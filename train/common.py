"""Shared utilities for training-loop drivers.

Used by both:
  - ``train/run_training.py``                  (in-process Python self-play)
  - ``train/run_training_chess_rust.py``       (Rust self-play subprocess; Phase 5)

What lives here:
  - Console display helpers (durations, section headers, rolling phase timer)
  - Checkpoint I/O (save, scan-and-load latest, load-at-path)
  - Arena evaluation (head-to-head MCTS play, opening-temperature sampling)
  - Default thresholds used by both drivers

What does **not** live here: preset config dicts (driver-specific), the
game registry, and the orchestration loop itself.
"""

import collections
import glob
import os

import numpy as np
import torch
import torch.optim as optim

from mcts.mcts import MCTS
from model.model import ResNet

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_ARENA_THRESHOLD = 0.55
# Sample the first N moves of arena games proportionally to the MCTS visit
# policy. Without this both networks play deterministically from a fixed start
# and every eval game collapses to the same line, which is why we ever saw
# "multiples of 10" results before this knob was added.
DEFAULT_EVAL_OPENING_TEMP_MOVES = 2
DEFAULT_CHECKPOINT_DIR = "checkpoints"

W = 70  # console width


# ── Display helpers ───────────────────────────────────────────────────────────


def fmt_time(s):
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


def hr(char="─"):
    return char * W


def section(title, char="─"):
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


def ckpt_prefix(args):
    """Checkpoint filename prefix derived from the game name (e.g., 'chess_iter_')."""
    return f"{args['game']}_iter_"


def champion_path(checkpoint_dir, args):
    """Path of the arena-gated 'best so far' checkpoint for this game."""
    return os.path.join(checkpoint_dir, f"{args['game']}_best.pt")


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
    files = sorted(glob.glob(os.path.join(checkpoint_dir, f"{ckpt_prefix(args)}*.pt")))
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


def load_model_at(path, game, args):
    """Load a ResNet from a checkpoint path, in eval mode."""
    ckpt = torch.load(path, weights_only=False)
    model = ResNet(
        game, num_res_blocks=args["num_res_blocks"], num_hidden=args["num_hidden"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────


def play_eval_game(game, mcts_white, mcts_black, max_moves, opening_temp_moves=0):
    """Play one eval game. Returns 1 (white wins), -1 (black wins), 0 (draw).

    The first `opening_temp_moves` moves are sampled proportionally to the
    MCTS visit policy so games starting from the same position diverge;
    remaining moves are played greedily (argmax) to measure best-play strength.
    """
    state = game.get_initial_state()
    player = 1
    mcts_white._root = None
    mcts_black._root = None

    for move_count in range(max_moves):
        mcts = mcts_white if player == 1 else mcts_black
        policy = mcts.search(state, player)
        if move_count < opening_temp_moves:
            action = int(np.random.choice(game.action_size, p=policy))
        else:
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
    opening_temp_moves = args.get(
        "eval_opening_temp_moves", DEFAULT_EVAL_OPENING_TEMP_MOVES
    )
    wins = draws = losses = 0

    for i in range(n):
        mcts_new = MCTS(game, model=new_model, num_searches=searches)
        mcts_old = MCTS(game, model=old_model, num_searches=searches)
        new_is_white = i % 2 == 0
        if new_is_white:
            result = play_eval_game(
                game, mcts_new, mcts_old, args["max_moves"], opening_temp_moves
            )
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1
        else:
            result = play_eval_game(
                game, mcts_old, mcts_new, args["max_moves"], opening_temp_moves
            )
            if result == -1:
                wins += 1
            elif result == 1:
                losses += 1
            else:
                draws += 1

    win_rate = (wins + 0.5 * draws) / n if n > 0 else 0.0
    return {"wins": wins, "draws": draws, "losses": losses, "win_rate": win_rate}
