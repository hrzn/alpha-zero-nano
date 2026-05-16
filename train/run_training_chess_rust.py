"""Chess training driver backed by the Rust self-play binary.

Phase 5 of `design/RUST_PORT_PLAN.md`. Per iteration:

  1. Export the current PyTorch model to ONNX in-process.
  2. Subprocess `rust/target/release/selfplay` to generate K games into
     `<run_dir>/shards/iter_NNNN/{states,policies,values}.npy`.
  3. Load the new shards into the Python-side replay buffer.
  4. Train for E epochs of full buffer passes via `train.train::train_step`.
  5. Save a `.pt` checkpoint.
  6. Periodically run arena evaluation against the current champion
     (Python-side PyTorch, mirrors `train/run_training.py`).

The Rust binary owns all self-play wall-clock; Python owns gradient updates,
checkpoints, and arena gating. This script is intentionally separate from
`train/run_training.py` so the original C4/tictactoe driver stays untouched.

Usage (run from repo root):
    uv run python -m train.run_training_chess_rust                       # S preset
    uv run python -m train.run_training_chess_rust --preset XS
    uv run python -m train.run_training_chess_rust --preset M --device mps
    uv run python -m train.run_training_chess_rust --dir runs/my_run

Resumes from the latest `.pt` checkpoint in the run dir if one is present.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from chess_game.chess_game import ChessGame
from model.model import ResNet
from train.common import (
    DEFAULT_ARENA_THRESHOLD,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_EVAL_OPENING_TEMP_MOVES,
    PhaseTimer,
    W,
    champion_path,
    ckpt_prefix,
    fmt_time,
    hr,
    load_latest_checkpoint,
    load_model_at,
    run_evaluation,
    save_checkpoint,
    section,
)
from train.shards import load_shard
from train.train import train_step

# ── Presets ────────────────────────────────────────────────────────────────────

# Chess-only. All run the Rust selfplay binary; Python handles training.
# Reuse field names from `train/run_training.py` where they map cleanly so
# arena/checkpoint helpers in `train.common` work unchanged.
PRESETS = {
    "XS": {
        "_description": "Tiny end-to-end check — 1 minute total, no GPU",
        "game": "chess",
        # Model — must match the architecture inside the saved checkpoints.
        "num_res_blocks": 3,
        "num_hidden": 64,
        # MCTS / selfplay (Rust side)
        "num_searches": 50,
        "selfplay_batch_size": 8,
        "c_puct": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 4,
        "num_workers": 2,
        "max_moves": 40,
        "temp_threshold": 8,
        # Training (Python side)
        "num_epochs": 2,
        "train_batch_size": 32,
        "lr": 1e-3,
        "lr_milestones": [],
        "replay_buffer_size": 2_000,
        # Loop
        "num_iterations": 5,
        "checkpoint_interval": 2,
        "eval_interval": 5,
        "eval_games": 4,
        "eval_searches": 25,
    },
    "S": {
        "_description": "Small chess training — overnight target on M1",
        "game": "chess",
        "num_res_blocks": 5,
        "num_hidden": 128,
        "num_searches": 200,
        "selfplay_batch_size": 32,
        "c_puct": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 40,
        "num_workers": 4,
        "max_moves": 120,
        "temp_threshold": 25,
        "num_epochs": 4,
        "train_batch_size": 256,
        "lr": 1e-3,
        "lr_milestones": [50],
        "replay_buffer_size": 30_000,
        "num_iterations": 100,
        "checkpoint_interval": 5,
        "eval_interval": 5,
        "eval_games": 10,
        "eval_searches": 100,
    },
    "M": {
        "_description": "Full chess run — hours-to-days; non-trivial play target",
        "game": "chess",
        "num_res_blocks": 10,
        "num_hidden": 256,
        "num_searches": 400,
        "selfplay_batch_size": 64,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 80,
        "num_workers": 8,
        "max_moves": 200,
        "temp_threshold": 30,
        "num_epochs": 5,
        "train_batch_size": 512,
        "lr": 1e-3,
        "lr_milestones": [75, 150],
        "replay_buffer_size": 80_000,
        "num_iterations": 200,
        "checkpoint_interval": 5,
        "eval_interval": 10,
        "eval_games": 20,
        "eval_searches": 200,
        "arena_threshold": 0.55,
    },
}

REPO_ROOT = Path(__file__).resolve().parent.parent
RUST_BINARY = REPO_ROOT / "rust" / "target" / "release" / "selfplay"


# ── Helpers ───────────────────────────────────────────────────────────────────


def ensure_rust_binary() -> Path:
    """Build the release `selfplay` binary if it isn't already present.
    Returns its path. The Phase 5 driver is the only place this build is
    triggered — fast subsequent invocations rely on the cache."""
    if RUST_BINARY.exists():
        return RUST_BINARY
    print("  Building rust/target/release/selfplay (one-time)...")
    rust_dir = REPO_ROOT / "rust"
    subprocess.run(
        ["cargo", "build", "--release", "--bin", "selfplay"],
        cwd=rust_dir,
        check=True,
    )
    if not RUST_BINARY.exists():
        raise FileNotFoundError(
            f"cargo build succeeded but {RUST_BINARY} missing — check the cargo output"
        )
    return RUST_BINARY


def find_libonnxruntime() -> str:
    """Locate the libonnxruntime dylib shipped with the project's
    `onnxruntime` Python package. The Rust binary's auto-discovery walks
    up to find this, but passing it explicitly via env is more robust."""
    candidate = (
        REPO_ROOT
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "onnxruntime"
        / "capi"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"onnxruntime not installed in venv ({candidate}). "
            "Run `uv sync --group dev`."
        )
    matches = sorted(candidate.glob("libonnxruntime*.dylib")) + sorted(
        candidate.glob("libonnxruntime*.so")
    )
    if not matches:
        raise FileNotFoundError(
            f"no libonnxruntime dylib in {candidate}"
        )
    return str(matches[0])


def export_pytorch_model_to_onnx(model: ResNet, out_path: Path, opset: int = 17) -> None:
    """In-process ONNX export of a live PyTorch ResNet.

    Reuses the `ChessForward` wrapper from `tools/export_chess_onnx.py` so
    the value output is shape `(B, 1)` and matches Rust's loader. Toggles
    eval mode (BN must use running stats) and restores training mode after.
    """
    # Local import — the export tool lives under tools/ which isn't a package
    # (no __init__.py). Add it to sys.path on first use.
    tools_dir = str(REPO_ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from export_chess_onnx import ChessForward, export

    was_training = model.training
    wrapper = ChessForward(model)
    wrapper.eval()  # propagates to inner ResNet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # The wrapper holds a reference to `model`; we need to export from CPU
    # because torch.onnx.export defaults to the model's current device and
    # some backends (MPS) don't yet support the ONNX exporter cleanly.
    # Move temporarily, export, restore.
    original_device = next(model.parameters()).device
    if original_device.type != "cpu":
        wrapper.cpu()
    try:
        export(wrapper, str(out_path), opset=opset)
    finally:
        if original_device.type != "cpu":
            model.to(original_device)
        if was_training:
            model.train()


def run_rust_selfplay(
    binary: Path,
    onnx_path: Path,
    out_dir: Path,
    args: dict,
    iteration: int,
    seed_base: int = 0,
) -> None:
    """Invoke the Rust selfplay binary and stream its stdout to ours."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--model", str(onnx_path),
        "--out", str(out_dir),
        "--num-games", str(args["num_self_play_games"]),
        "--num-workers", str(args["num_workers"]),
        "--num-searches", str(args["num_searches"]),
        "--c-puct", str(args["c_puct"]),
        "--batch-size", str(args["selfplay_batch_size"]),
        "--max-moves", str(args["max_moves"]),
        "--temp-threshold", str(args["temp_threshold"]),
        "--dirichlet-alpha", str(args["dirichlet_alpha"]),
        "--dirichlet-epsilon", str(args["dirichlet_epsilon"]),
        "--seed", str(seed_base + iteration),
    ]
    env = os.environ.copy()
    env["ORT_DYLIB_PATH"] = find_libonnxruntime()
    # Stream stdout so the user sees Rust's progress lines live.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"  [rust] {line.rstrip()}")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"selfplay binary exited with rc={rc}")


# ── Training loop ────────────────────────────────────────────────────────────


def run_training(preset_name: str = "S", device: str | None = None,
                 checkpoint_dir: str | None = None) -> tuple[ResNet, list]:
    """Main training loop. Returns (final_model, loss_history)."""
    args = dict(PRESETS[preset_name])
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(
            DEFAULT_CHECKPOINT_DIR, f"chess_rust_{preset_name.lower()}"
        )
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect device for the *training* model. The Rust selfplay binary
    # is CPU-only (ort/CoreML/CUDA TBD), independent of this choice.
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    game = ChessGame()
    binary = ensure_rust_binary()

    # ── Header ───────────────────────────────────────────────────────────────
    print(hr("═"))
    print(f"  AlphaZero Chess (rust selfplay) — {preset_name} preset")
    print(f"  {args['_description']}")
    print(f"  Model  : {args['num_res_blocks']} res_blocks × {args['num_hidden']} hidden")
    print(
        f"  MCTS   : {args['num_searches']} sims  c_puct={args['c_puct']}"
        f"  dir_α={args['dirichlet_alpha']}  dir_ε={args['dirichlet_epsilon']}"
    )
    print(
        f"  Play   : {args['num_self_play_games']} games/iter  "
        f"{args['num_workers']} workers  batch={args['selfplay_batch_size']}  "
        f"max_moves={args['max_moves']}  temp→0 after move {args['temp_threshold']}"
    )
    print(
        f"  Train  : {args['num_epochs']} epochs/iter  lr={args['lr']}  "
        f"buf≤{args['replay_buffer_size']:,}"
    )
    arena_threshold = args.get("arena_threshold", DEFAULT_ARENA_THRESHOLD)
    opening_temp = args.get("eval_opening_temp_moves", DEFAULT_EVAL_OPENING_TEMP_MOVES)
    print(
        f"  Arena  : {args['eval_games']} games vs champion; "
        f"promote if win rate > {arena_threshold:.0%}  (opening temp moves={opening_temp})"
    )
    print(f"  Device : {device} (training)  |  dir: {checkpoint_dir}")
    print(f"  Binary : {binary}")
    print(hr("═"))

    # ── Resume or fresh start ─────────────────────────────────────────────────
    model, optimizer, start_iter, loss_history = load_latest_checkpoint(
        str(checkpoint_dir), game, args
    )
    if model is None:
        model = ResNet(
            game,
            num_res_blocks=args["num_res_blocks"],
            num_hidden=args["num_hidden"],
        )
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        start_iter = 0
        loss_history = []
        save_checkpoint(
            str(checkpoint_dir / f"{ckpt_prefix(args)}0000.pt"),
            0, model, optimizer, loss_history, args,
        )
        save_checkpoint(
            champion_path(str(checkpoint_dir), args),
            0, model, optimizer, loss_history, args,
        )
        print("  Fresh start. Saved iter-0 baseline and initial champion.")
    else:
        champ_p = Path(champion_path(str(checkpoint_dir), args))
        if champ_p.exists():
            champ_iter = torch.load(champ_p, weights_only=False).get("iteration", "?")
            print(f"  Resumed from iter {start_iter}; champion = iter {champ_iter}.")
        else:
            save_checkpoint(
                str(champ_p), start_iter, model, optimizer, loss_history, args
            )
            print(f"  Resumed from iter {start_iter}; seeded champion at iter {start_iter}.")

    model = model.to(device)

    # Apply LR milestones already passed (resume case).
    current_lr = args["lr"]
    for m in args["lr_milestones"]:
        if start_iter >= m:
            current_lr *= 0.5
    if current_lr != args["lr"]:
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr
        print(f"  LR restored to {current_lr:.2e} (milestones {args['lr_milestones']} passed).")

    # Per-iteration artifact dirs.
    shards_root = checkpoint_dir / "shards"
    onnx_dir = checkpoint_dir / "onnx"
    shards_root.mkdir(exist_ok=True)
    onnx_dir.mkdir(exist_ok=True)

    replay_buffer: collections.deque = collections.deque(maxlen=args["replay_buffer_size"])
    timer = PhaseTimer(window=20)

    # ── Main loop ─────────────────────────────────────────────────────────────
    for iteration in range(start_iter, args["num_iterations"]):
        iter_num = iteration + 1
        ts = datetime.now().strftime("%H:%M:%S")
        print()
        print(section(f"Iter {iter_num}/{args['num_iterations']}  [{preset_name}]  {ts}"))

        if iteration in args["lr_milestones"]:
            current_lr *= 0.5
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr
            print(f"  LR decay → {current_lr:.2e}")

        # ── Export current model to ONNX ──────────────────────────────────
        t0 = time.perf_counter()
        onnx_path = onnx_dir / f"iter_{iter_num:04d}.onnx"
        export_pytorch_model_to_onnx(model, onnx_path)
        export_time = time.perf_counter() - t0
        timer.record("export", export_time)

        # ── Rust self-play ───────────────────────────────────────────────
        t0 = time.perf_counter()
        shard_dir = shards_root / f"iter_{iter_num:04d}"
        run_rust_selfplay(binary, onnx_path, shard_dir, args, iter_num)
        sp_time = time.perf_counter() - t0
        timer.record("self_play", sp_time)

        # Load shards into the replay buffer.
        shard = load_shard(str(shard_dir), mmap=False)
        n_new = len(shard)
        for i in range(n_new):
            replay_buffer.append(
                (shard.states[i], shard.policies[i], float(shard.values[i]))
            )
        print(
            f"  Selfplay  +{n_new:,} ex  buf {len(replay_buffer):,}/"
            f"{args['replay_buffer_size']:,}  [export {fmt_time(export_time)}  "
            f"selfplay {fmt_time(sp_time)}]"
        )

        # ── Training ─────────────────────────────────────────────────────
        t0 = time.perf_counter()
        buf_list = list(replay_buffer)
        encoded_states = np.array([e[0] for e in buf_list], dtype=np.float32)
        policies = np.array([e[1] for e in buf_list], dtype=np.float32)
        outcomes = np.array([e[2] for e in buf_list], dtype=np.float32)

        model.train()
        epoch_losses: list[float] = []
        bsz = args["train_batch_size"]
        n_steps = 0
        n_epochs = args["num_epochs"]
        buf_size = len(buf_list)
        for epoch_i in range(n_epochs):
            perm = np.random.permutation(buf_size)
            ep_losses: list[float] = []
            for start in range(0, buf_size, bsz):
                idx = perm[start : start + bsz]
                if len(idx) < max(1, bsz // 2):
                    continue
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
                f"\r  Training  epoch {epoch_i+1}/{n_epochs}  step {n_steps}  "
                f"loss={ep_loss:.4f}  {fmt_time(time.perf_counter() - t0):>6}   ",
                end="",
                flush=True,
            )

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        loss_history.append(avg_loss)
        train_time = time.perf_counter() - t0
        timer.record("train", train_time)

        trend = ""
        if len(loss_history) >= 2:
            delta = loss_history[-1] - loss_history[-2]
            trend = " ↓" if delta < -0.005 else (" ↑" if delta > 0.005 else " →")
        print(
            f"\r  Training  {n_epochs} epochs  {n_steps} steps  batch={bsz}  "
            f"loss={avg_loss:.4f}{trend}  [{fmt_time(train_time)}]          "
        )

        # ── Iteration summary ────────────────────────────────────────────
        iter_total = export_time + sp_time + train_time
        timer.record("iter", iter_total)
        iters_left = args["num_iterations"] - iter_num
        eta = timer.mean("iter") * iters_left
        print(
            f"  {hr('·')[:W-2]}\n"
            f"  Total {fmt_time(iter_total)}  "
            f"(sp {fmt_time(timer.mean('self_play'))} avg  "
            f"tr {fmt_time(timer.mean('train'))} avg)  │  ETA {fmt_time(eta)}"
        )

        # ── Checkpoint ───────────────────────────────────────────────────
        if iter_num % args["checkpoint_interval"] == 0:
            ckpt_path = checkpoint_dir / f"{ckpt_prefix(args)}{iter_num:04d}.pt"
            save_checkpoint(str(ckpt_path), iter_num, model, optimizer, loss_history, args)
            with open(checkpoint_dir / "loss_history.json", "w") as f:
                json.dump(loss_history, f)
            print(f"  Saved {ckpt_path}")

        # ── Arena eval & gating ──────────────────────────────────────────
        if iter_num % args["eval_interval"] == 0:
            champ_p = Path(champion_path(str(checkpoint_dir), args))
            if champ_p.exists():
                champ_ckpt = torch.load(champ_p, weights_only=False)
                champ_iter = champ_ckpt.get("iteration", 0)
                print(
                    f"\n  {section(f'Arena: iter {iter_num} vs champion (iter {champ_iter})', char='┄')}"
                )
                old_model = load_model_at(str(champ_p), game, args).to(device)
                t0 = time.perf_counter()
                result = run_evaluation(game, model, old_model, args)
                eval_time = time.perf_counter() - t0
                timer.record("eval", eval_time)

                w, d, l = result["wins"], result["draws"], result["losses"]
                wr = result["win_rate"]
                promoted = wr > arena_threshold
                if promoted:
                    save_checkpoint(
                        str(champ_p), iter_num, model, optimizer, loss_history, args
                    )
                    verdict = f"↑ PROMOTED to champion (was iter {champ_iter})"
                else:
                    verdict = f"· champion held at iter {champ_iter}"
                print(
                    f"  {w}W {d}D {l}L  win rate {wr:.1%}  {verdict}  [{fmt_time(eval_time)}]"
                )

                eval_hist_path = checkpoint_dir / "eval_history.json"
                eval_hist: list = []
                if eval_hist_path.exists():
                    with open(eval_hist_path) as f:
                        eval_hist = json.load(f)
                eval_hist.append(
                    {
                        "iteration": iter_num,
                        "champion_iter": champ_iter,
                        "wins": w, "draws": d, "losses": l,
                        "win_rate": wr, "promoted": promoted,
                    }
                )
                with open(eval_hist_path, "w") as f:
                    json.dump(eval_hist, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print(hr("═"))
    print(f"  Training complete — {preset_name} preset  {args['num_iterations']} iterations")
    if loss_history:
        print(f"  Loss   : {loss_history[0]:.4f} → {loss_history[-1]:.4f}")
    print(
        f"  Timing : self-play {fmt_time(timer.mean('self_play'))} avg  │  "
        f"training {fmt_time(timer.mean('train'))} avg  │  "
        f"total {fmt_time(timer.mean('iter'))} avg/iter"
    )
    if timer.mean("eval") > 0:
        print(f"  Eval   : {fmt_time(timer.mean('eval'))} avg per eval round")
    print(f"  Output : {checkpoint_dir}")
    print(hr("═"))

    return model, loss_history


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AlphaZero Chess training driver — Rust self-play backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {name}: {p['_description']}" for name, p in PRESETS.items()
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
        help="Compute device for training (default: auto)",
    )
    parser.add_argument(
        "--dir",
        dest="checkpoint_dir",
        default=None,
        help=f"Run directory (default: {DEFAULT_CHECKPOINT_DIR}/chess_rust_<preset>)",
    )
    return parser.parse_args()


def main() -> None:
    a = parse_args()
    run_training(preset_name=a.preset, device=a.device, checkpoint_dir=a.checkpoint_dir)


if __name__ == "__main__":
    main()
