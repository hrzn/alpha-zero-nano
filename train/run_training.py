"""Standalone training script with preset configurations.

Presets (chess):
  XS — tiny model, for debugging; ~10s/iter, no GPU needed
  S  — small model; first signs of non-random play expected by iter 20-30
  M  — full training run; hours on M1 + MPS; target: non-trivial chess play

Presets (connect 4):
  C4 — small model; should clearly beat random within a few dozen iterations

Usage (run as a module from the project root, not as a script — running it
as `python train/run_training.py` shadows the `train` package with the
script's directory and breaks the `from train.train import …` import):
    uv run python -m train.run_training                         # S preset, auto device
    uv run python -m train.run_training --preset XS
    uv run python -m train.run_training --preset M --device mps
    uv run python -m train.run_training --preset C4
    uv run python -m train.run_training --preset S --dir runs/my_run

Resumes automatically from the latest checkpoint in --dir.
"""

import argparse
import collections
import glob
import json
import multiprocessing
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from chess_game.chess_game import ChessGame
from connect4 import Connect4
from mcts.mcts import MCTS
from model.model import ResNet
from train.train import _worker_self_play, self_play, train_step

# Game registry: preset's "game" field → (class, display name)
_GAMES = {
    "chess": (ChessGame, "Chess"),
    "connect4": (Connect4, "Connect 4"),
}

# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS = {
    "XS": {
        "_description": "Tiny model — verify pipeline end-to-end (~10s/iter)",
        "game": "chess",
        # Model
        "num_res_blocks": 3,
        "num_hidden": 64,
        # MCTS
        "num_searches": 100,
        "mcts_batch_size": 5,
        "c_puct": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        # Self-play
        "num_self_play_games": 2,
        "n_workers": 1,  # sequential; enables mcts_batch_size (Opt 4)
        "max_moves": 30,
        "temp_threshold": 8,  # argmax after move 8
        # Training — each epoch = one full pass over the replay buffer
        "num_epochs": 3,
        "train_batch_size": 32,
        "lr": 1e-3,
        "lr_milestones": [],  # iterations at which to halve LR
        "replay_buffer_size": 1_000,
        # Loop
        "num_iterations": 50,
        "checkpoint_interval": 2,
        "eval_interval": 2,
        "eval_games": 4,
        "eval_searches": 10,
    },
    "S": {
        "_description": "Small model — first non-random play expected by iter 20-30",
        "game": "chess",
        "num_res_blocks": 5,
        "num_hidden": 128,
        "num_searches": 200,
        "mcts_batch_size": 40,  # 5 batches/worker — Opts 3+4 combined
        "c_puct": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 20,
        "n_workers": 4,
        "max_moves": 100,
        "temp_threshold": 20,  # argmax after move 20
        # Training — each epoch = one full pass over the replay buffer
        "num_epochs": 4,
        "train_batch_size": 256,
        "lr": 1e-3,
        "lr_milestones": [50],
        "replay_buffer_size": 20_000,
        "num_iterations": 100,
        "checkpoint_interval": 5,
        "eval_interval": 5,
        "eval_games": 10,
        "eval_searches": 100,
    },
    "M": {
        "_description": "Full run — hours on M1 + MPS; target: non-trivial chess play",
        "game": "chess",
        "num_res_blocks": 10,
        "num_hidden": 256,
        "num_searches": 400,
        "mcts_batch_size": 80,  # 5 batches/worker — Opts 3+4 combined
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 50,
        "n_workers": 8,
        "max_moves": 200,
        "temp_threshold": 30,  # argmax after move 30
        # Training — each epoch = one full pass over the replay buffer
        "num_epochs": 5,
        "train_batch_size": 512,
        "lr": 1e-3,
        "lr_milestones": [75, 150],
        "replay_buffer_size": 50_000,
        "num_iterations": 200,
        "checkpoint_interval": 5,
        "eval_interval": 10,
        "eval_games": 20,
        "eval_searches": 200,
    },
    "C4": {
        "_description": "Connect 4 — should clearly beat random within ~30 iters",
        "game": "connect4",
        "num_res_blocks": 3,
        "num_hidden": 64,
        "num_searches": 100,
        "mcts_batch_size": 20,
        "c_puct": 1.0,
        "dirichlet_alpha": 1.0,  # 7-action space → larger alpha than chess
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 30,
        "n_workers": 4,
        "max_moves": 42,  # board size; games end naturally well before this
        "temp_threshold": 8,
        "num_epochs": 4,
        "train_batch_size": 128,
        "lr": 1e-3,
        "lr_milestones": [],
        "replay_buffer_size": 10_000,
        "num_iterations": 50,
        "checkpoint_interval": 5,
        "eval_interval": 5,
        "eval_games": 20,
        "eval_searches": 50,
    },
}

DEFAULT_CHECKPOINT_DIR = "checkpoints"
W = 70  # console width


# ── Display helpers ───────────────────────────────────────────────────────────


def _fmt_time(s):
    """Human-readable duration: 450ms / 1.2s / 2m03s / 1h04m."""
    if s < 1.0:
        return f"{s * 1000:.0f}ms"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _hr(char="─"):
    return char * W


def _section(title, char="─"):
    pad = W - len(title) - 4
    return f"── {title} " + char * max(0, pad)


class PhaseTimer:
    """Rolling-window timing tracker for named phases."""

    def __init__(self, window=20):
        self._data = {}
        self._window = window

    def record(self, phase, elapsed):
        if phase not in self._data:
            self._data[phase] = collections.deque(maxlen=self._window)
        self._data[phase].append(elapsed)

    def mean(self, phase):
        d = self._data.get(phase, [])
        return sum(d) / len(d) if d else 0.0

    def last(self, phase):
        d = self._data.get(phase)
        return d[-1] if d else 0.0


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def save_checkpoint(path, iteration, model, optimizer, loss_history, args):
    """Save training state.

    Checkpoint keys: iteration, model_state_dict, optimizer_state_dict,
                     loss_history, args.
    """
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "args": {k: v for k, v in args.items() if not k.startswith("_")},
        },
        path,
    )


def load_latest_checkpoint(checkpoint_dir, game, args):
    """Scan checkpoint_dir for <game>_iter_*.pt and load the highest-numbered one.

    Returns (model, optimizer, iteration, loss_history).
    If no checkpoint is found, returns (None, None, None, []).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(checkpoint_dir, f"{_ckpt_prefix(args)}*.pt")))
    if not files:
        return None, None, None, []

    ckpt = torch.load(files[-1], weights_only=False)
    model = ResNet(
        game, num_res_blocks=args["num_res_blocks"], num_hidden=args["num_hidden"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return model, optimizer, ckpt["iteration"], ckpt.get("loss_history", [])


def _load_model_at(path, game, args):
    """Load a ResNet from a checkpoint path, in eval mode."""
    ckpt = torch.load(path, weights_only=False)
    model = ResNet(
        game, num_res_blocks=args["num_res_blocks"], num_hidden=args["num_hidden"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _ckpt_prefix(args):
    """Checkpoint filename prefix derived from the game name (e.g., 'chess_iter_')."""
    return f"{args['game']}_iter_"


def _find_eval_baseline(checkpoint_dir, current_iter, eval_interval, args):
    """Return the path of the most recent checkpoint at least eval_interval iters behind current.

    Returns None if no such checkpoint exists yet.
    """
    target = current_iter - eval_interval
    prefix = _ckpt_prefix(args)
    files = sorted(glob.glob(os.path.join(checkpoint_dir, f"{prefix}*.pt")))
    best = None
    for f in files:
        stem = os.path.basename(f)
        try:
            it = int(stem[len(prefix) : -len(".pt")])
        except ValueError:
            continue
        if it <= target:
            best = f
    return best


# ── Evaluation ────────────────────────────────────────────────────────────────


def _play_eval_game(game, mcts_white, mcts_black, max_moves):
    """Play one greedy game. Returns 1 (white wins), -1 (black wins), 0 (draw)."""
    state = game.get_initial_state()
    player = 1
    mcts_white._root = None
    mcts_black._root = None

    for _ in range(max_moves):
        mcts = mcts_white if player == 1 else mcts_black
        policy = mcts.search(state, player)
        action = int(np.argmax(policy))
        # Both trees advance so tree reuse stays valid for both sides
        mcts_white.advance_root(action)
        mcts_black.advance_root(action)
        state = game.update_state(state, action, player)
        value, terminated = game.get_value_and_terminated(state, action)
        if terminated:
            return 0 if value == 0 else player  # player who just moved won
        player = game.get_opponent(player)

    return 0  # draw by move limit


def run_evaluation(game, new_model, old_model, args):
    """Play eval_games between new_model and old_model, alternating colours.

    Returns dict: wins, draws, losses, win_rate — all from new_model's perspective.
    """
    n = args["eval_games"]
    searches = args["eval_searches"]
    wins = draws = losses = 0

    for i in range(n):
        mcts_new = MCTS(game, model=new_model, num_searches=searches)
        mcts_old = MCTS(game, model=old_model, num_searches=searches)
        new_is_white = i % 2 == 0
        if new_is_white:
            result = _play_eval_game(game, mcts_new, mcts_old, args["max_moves"])
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        else:
            result = _play_eval_game(game, mcts_old, mcts_new, args["max_moves"])
            if result == -1:
                wins += 1
            elif result == 1:
                losses += 1
            else:
                draws += 1

    win_rate = (wins + 0.5 * draws) / n if n > 0 else 0.0
    return {"wins": wins, "draws": draws, "losses": losses, "win_rate": win_rate}


# ── Training loop ─────────────────────────────────────────────────────────────


def run_training(preset_name="S", device=None, checkpoint_dir=None):
    """Main training loop.

    Args:
        preset_name: one of "XS", "S", "L"
        device:      "cpu" / "mps" / "cuda" or None for auto-detect
        checkpoint_dir: path for checkpoints (default: checkpoints/<preset>)
    """
    args = dict(PRESETS[preset_name])
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(DEFAULT_CHECKPOINT_DIR, preset_name.lower())

    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    game_cls, game_display = _GAMES[args["game"]]
    game = game_cls()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Header ───────────────────────────────────────────────────────────────
    print(_hr("═"))
    print(f"  AlphaZero {game_display} — {preset_name} preset")
    print(f"  {args['_description']}")
    print(
        f"  Model  : {args['num_res_blocks']} res_blocks × {args['num_hidden']} hidden"
    )
    print(
        f"  MCTS   : {args['num_searches']} searches  c_puct={args['c_puct']}"
        f"  dir_α={args['dirichlet_alpha']}  dir_ε={args['dirichlet_epsilon']}"
    )
    if args["n_workers"] > 1:
        sp_mode = f"{args['n_workers']} workers  batch_size={args['mcts_batch_size']} (Opts 3+4)"
    else:
        sp_mode = f"sequential  batch_size={args['mcts_batch_size']} (Opt 4)"
    temp_str = (
        f"  temp→0 after move {args['temp_threshold']}"
        if args.get("temp_threshold")
        else ""
    )
    print(
        f"  Play   : {args['num_self_play_games']} games/iter  {sp_mode}  max_moves={args['max_moves']}{temp_str}"
    )
    print(f"  Value bootstrap at move cap (no flat-draw signal)")
    print(
        f"  Train  : {args['num_epochs']} epochs/iter (full buffer pass)  lr={args['lr']}  buf≤{args['replay_buffer_size']:,}"
    )
    milestones = args["lr_milestones"]
    lr_str = f"  LR ×0.5 at iters {milestones}" if milestones else "  LR: constant"
    print(lr_str)
    print(
        f"  Loop   : {args['num_iterations']} iters  ckpt every {args['checkpoint_interval']}  eval every {args['eval_interval']}"
    )
    print(f"  Device : {device}  |  dir: {checkpoint_dir}")
    print(_hr("═"))

    # ── Resume or fresh start ─────────────────────────────────────────────────
    model, optimizer, start_iter, loss_history = load_latest_checkpoint(
        checkpoint_dir, game, args
    )
    if model is None:
        model = ResNet(
            game, num_res_blocks=args["num_res_blocks"], num_hidden=args["num_hidden"]
        )
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        start_iter = 0
        loss_history = []
        # Save iter-0 baseline so evaluation has something to compare against
        save_checkpoint(
            os.path.join(checkpoint_dir, f"{_ckpt_prefix(args)}0000.pt"),
            0,
            model,
            optimizer,
            loss_history,
            args,
        )
        print(f"  Fresh start. Saved iter-0 baseline.")
    else:
        print(f"  Resumed from iteration {start_iter}.")

    model = model.to(device)

    # Reapply LR milestones already passed (resume case)
    current_lr = args["lr"]
    for m in args["lr_milestones"]:
        if start_iter >= m:
            current_lr *= 0.5
    if current_lr != args["lr"]:
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        print(
            f"  LR restored to {current_lr:.2e} (milestones {args['lr_milestones']} passed)."
        )

    mcts = MCTS(
        game,
        model=model,
        num_searches=args["num_searches"],
        c_puct=args["c_puct"],
        batch_size=args["mcts_batch_size"],
        dirichlet_alpha=args["dirichlet_alpha"],
        dirichlet_epsilon=args["dirichlet_epsilon"],
    )

    replay_buffer = collections.deque(maxlen=args["replay_buffer_size"])
    timer = PhaseTimer(window=20)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(start_iter, args["num_iterations"]):
        iter_num = iteration + 1
        ts = datetime.now().strftime("%H:%M:%S")
        print()
        print(
            _section(f"Iter {iter_num}/{args['num_iterations']}  [{preset_name}]  {ts}")
        )

        # LR milestone
        if iteration in args["lr_milestones"]:
            current_lr *= 0.5
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            print(f"  LR decay → {current_lr:.2e}")

        # ── Self-play ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        n_games = args["num_self_play_games"]
        sp_task = (
            type(game),
            {k: v.cpu() for k, v in model.state_dict().items()},
            model.num_res_blocks,
            model.num_hidden,
            mcts.num_searches,
            mcts.c_puct,
            mcts.batch_size,
            mcts.dirichlet_alpha,
            mcts.dirichlet_epsilon,
            args.get("temp_threshold"),
            args["max_moves"],
        )
        ng_w = len(str(n_games))  # width for game counter

        all_game_examples = []
        move_counts = []

        def _sp_progress():
            n_done = len(move_counts)
            elapsed = time.perf_counter() - t0
            avg = f"{sum(move_counts)/n_done:.0f}" if n_done else "?"
            bar_w = 14
            filled = int(bar_w * n_done / n_games)
            bar = "█" * filled + "░" * (bar_w - filled)
            print(
                f"\r  Self-play  [{bar}] {n_done:{ng_w}}/{n_games}"
                f"  avg {avg} moves  {_fmt_time(elapsed):>6}   ",
                end="",
                flush=True,
            )

        if args["n_workers"] > 1:
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(args["n_workers"]) as pool:
                for examples in pool.imap_unordered(
                    _worker_self_play, [sp_task] * n_games
                ):
                    all_game_examples.append(examples)
                    move_counts.append(len(examples))
                    _sp_progress()
        else:
            for _ in range(n_games):
                examples = self_play(
                    game,
                    mcts,
                    max_moves=args["max_moves"],
                    temp_threshold=args.get("temp_threshold"),
                )
                all_game_examples.append(examples)
                move_counts.append(len(examples))
                _sp_progress()

        new_examples = [ex for g in all_game_examples for ex in g]
        sp_time = time.perf_counter() - t0
        timer.record("self_play", sp_time)
        replay_buffer.extend(new_examples)
        buf_size = len(replay_buffer)

        mc = sorted(move_counts)
        mc_mean = sum(mc) / len(mc)
        capped = sum(1 for m in move_counts if m >= args["max_moves"])
        cap_str = f"  {capped}/{n_games} capped" if capped else ""
        print(
            f"\r  Self-play   {n_games} games"
            f"  +{len(new_examples):,} ex  buf {buf_size:,}/{args['replay_buffer_size']:,}"
            f"  [{_fmt_time(sp_time)}]          "
        )
        print(
            f"  moves: {mc[0]}–{mc[-1]}"
            f"  median {mc[len(mc)//2]}  mean {mc_mean:.0f}{cap_str}"
        )

        # ── Training ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        buf_list = list(replay_buffer)
        encoded_states = np.array([e[0] for e in buf_list], dtype=np.float32)
        policies = np.array([e[1] for e in buf_list], dtype=np.float32)
        outcomes = np.array([e[2] for e in buf_list], dtype=np.float32)

        model.train()
        epoch_losses = []
        bsz = args["train_batch_size"]
        n_steps = 0
        n_epochs = args["num_epochs"]
        for epoch_i in range(n_epochs):
            perm = np.random.permutation(buf_size)
            ep_losses = []
            for start in range(0, buf_size, bsz):
                idx = perm[start : start + bsz]
                if len(idx) < max(1, bsz // 2):
                    continue  # skip tiny tail batch
                loss = train_step(
                    model,
                    optimizer,
                    (encoded_states[idx], policies[idx], outcomes[idx]),
                )
                ep_losses.append(loss)
                epoch_losses.append(loss)
                n_steps += 1
            ep_loss = sum(ep_losses) / len(ep_losses) if ep_losses else 0.0
            print(
                f"\r  Training    epoch {epoch_i+1}/{n_epochs}"
                f"  step {n_steps}  loss={ep_loss:.4f}"
                f"  {_fmt_time(time.perf_counter() - t0):>6}   ",
                end="",
                flush=True,
            )

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        loss_history.append(avg_loss)
        train_time = time.perf_counter() - t0
        timer.record("train", train_time)

        loss_trend = ""
        if len(loss_history) >= 2:
            delta = loss_history[-1] - loss_history[-2]
            loss_trend = " ↓" if delta < -0.005 else (" ↑" if delta > 0.005 else " →")
        print(
            f"\r  Training    {n_epochs} epochs  {n_steps} steps  batch={bsz}"
            f"  loss={avg_loss:.4f}{loss_trend}"
            f"  [{_fmt_time(train_time)}]          "
        )

        # ── Iteration summary ─────────────────────────────────────────────
        iter_total = sp_time + train_time
        timer.record("iter", iter_total)
        iters_left = args["num_iterations"] - iter_num
        eta = timer.mean("iter") * iters_left
        sp_avg = timer.mean("self_play")
        tr_avg = timer.mean("train")
        print(
            f"  {_hr('·')[:W-2]}\n"
            f"  Total {_fmt_time(iter_total)}"
            f"  (sp {_fmt_time(sp_avg)} avg  tr {_fmt_time(tr_avg)} avg)"
            f"  │  ETA {_fmt_time(eta)}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if iter_num % args["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{_ckpt_prefix(args)}{iter_num:04d}.pt")
            save_checkpoint(ckpt_path, iter_num, model, optimizer, loss_history, args)
            with open(os.path.join(checkpoint_dir, "loss_history.json"), "w") as f:
                json.dump(loss_history, f)
            print(f"  Saved {ckpt_path}")

        # ── Evaluation ────────────────────────────────────────────────────
        if iter_num % args["eval_interval"] == 0:
            baseline_path = _find_eval_baseline(
                checkpoint_dir, iter_num, args["eval_interval"], args
            )
            if baseline_path is not None:
                prefix = _ckpt_prefix(args)
                baseline_iter = int(
                    os.path.basename(baseline_path)[len(prefix) : -len(".pt")]
                )
                print(
                    f"\n  {_section(f'Eval: iter {iter_num} vs iter {baseline_iter}', char='┄')}"
                )
                old_model = _load_model_at(baseline_path, game, args).to(device)
                t0 = time.perf_counter()
                result = run_evaluation(game, model, old_model, args)
                eval_time = time.perf_counter() - t0
                timer.record("eval", eval_time)

                w, d, l = result["wins"], result["draws"], result["losses"]
                wr = result["win_rate"]
                trend = (
                    "▲ improving"
                    if wr > 0.55
                    else ("▼ regressing" if wr < 0.45 else "~ stable")
                )
                print(
                    f"  {w}W {d}D {l}L  win rate {wr:.1%}  {trend}"
                    f"  [{_fmt_time(eval_time)}]"
                )

                # Persist eval results so the notebook can plot win rate over time
                eval_hist_path = os.path.join(checkpoint_dir, "eval_history.json")
                eval_hist = []
                if os.path.exists(eval_hist_path):
                    with open(eval_hist_path) as f:
                        eval_hist = json.load(f)
                eval_hist.append(
                    {
                        "iteration": iter_num,
                        "baseline_iter": baseline_iter,
                        "wins": w,
                        "draws": d,
                        "losses": l,
                        "win_rate": wr,
                    }
                )
                with open(eval_hist_path, "w") as f:
                    json.dump(eval_hist, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print(_hr("═"))
    print(
        f"  Training complete — {preset_name} preset  {args['num_iterations']} iterations"
    )
    if loss_history:
        print(f"  Loss   : {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
    print(
        f"  Timing : self-play {_fmt_time(timer.mean('self_play'))} avg"
        f"  │  training {_fmt_time(timer.mean('train'))} avg"
        f"  │  total {_fmt_time(timer.mean('iter'))} avg/iter"
    )
    if timer.mean("eval") > 0:
        print(f"  Eval   : {_fmt_time(timer.mean('eval'))} avg per eval round")
    print(f"  Output : {checkpoint_dir}")
    print(_hr("═"))

    return model, loss_history


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="AlphaZero training (game selected by preset)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name} [{p['game']}]: {p['_description']}"
            for name, p in PRESETS.items()
        ),
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default="S",
        help="Training preset (default: S)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "cuda"],
        default=None,
        help="Compute device (default: auto-detect mps > cuda > cpu)",
    )
    parser.add_argument(
        "--dir",
        dest="checkpoint_dir",
        default=None,
        help=f"Checkpoint directory (default: {DEFAULT_CHECKPOINT_DIR}/<preset>)",
    )
    return parser.parse_args()


def main():
    a = parse_args()
    run_training(preset_name=a.preset, device=a.device, checkpoint_dir=a.checkpoint_dir)


if __name__ == "__main__":
    main()
