"""Standalone chess training script with checkpointing.

Runs unattended for many hours with automatic save/resume.
All hyperparameters live in the ARGS dict below — no CLI flags needed.

Usage:
    uv run python train/run_training.py

Resume: on startup the script scans checkpoints/ for the highest-numbered
chess_iter_*.pt file and continues from there automatically.
"""

import glob
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from chess_game.chess_game import ChessGame
from mcts.mcts import MCTS
from model.model import ResNet
from train.train import self_play, train_step

# ── Hyperparameters ───────────────────────────────────────────────────────────
ARGS = {
    # Model architecture
    "num_res_blocks": 5,
    "num_hidden": 128,
    # MCTS
    "num_searches": 100,
    # Self-play
    "num_self_play_games": 15,
    "max_moves": 200,
    # Training
    "num_epochs": 4,
    "batch_size": 128,
    "lr": 1e-3,
    # Loop
    "num_iterations": 200,
    "checkpoint_interval": 5,  # save every N iterations
}

CHECKPOINT_DIR = "checkpoints"
# ─────────────────────────────────────────────────────────────────────────────


def save_checkpoint(path, iteration, model, optimizer, loss_history, args):
    """Save training state to path.

    Checkpoint format:
        {
            "iteration":          int,
            "model_state_dict":   OrderedDict,
            "optimizer_state_dict": dict,
            "loss_history":       [float, ...],
            "args":               dict,
        }
    """
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "args": args,
        },
        path,
    )


def load_latest_checkpoint(checkpoint_dir, game, args):
    """Scan checkpoint_dir for chess_iter_*.pt and load the highest-numbered one.

    Returns (model, optimizer, iteration, loss_history).
    If no checkpoint is found, returns (None, None, None, []).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    pattern = os.path.join(checkpoint_dir, "chess_iter_*.pt")
    files = sorted(glob.glob(pattern))

    if not files:
        return None, None, None, []

    latest = files[-1]
    checkpoint = torch.load(latest, weights_only=False)

    model = ResNet(
        game,
        num_res_blocks=args["num_res_blocks"],
        num_hidden=args["num_hidden"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=args["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint["iteration"], checkpoint["loss_history"]


def run_training(args=None, checkpoint_dir=None):
    """Main training loop. Exposed as a function so tests can call it directly."""
    if args is None:
        args = ARGS
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR

    game = ChessGame()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if one exists
    model, optimizer, start_iteration, loss_history = load_latest_checkpoint(
        checkpoint_dir, game, args
    )

    if model is None:
        model = ResNet(
            game,
            num_res_blocks=args["num_res_blocks"],
            num_hidden=args["num_hidden"],
        )
        optimizer = optim.Adam(model.parameters(), lr=args["lr"])
        start_iteration = 0
        loss_history = []
        print("Starting fresh training.")
    else:
        print(f"Resumed from iteration {start_iteration}.")

    mcts = MCTS(game, model=model, num_searches=args["num_searches"])

    for iteration in range(start_iteration, args["num_iterations"]):
        iter_start = datetime.now()

        # ── Self-play ──────────────────────────────────────────────────────
        examples = []
        for g in range(args["num_self_play_games"]):
            game_examples = self_play(game, mcts, max_moves=args["max_moves"])
            examples += game_examples

        # ── Train ──────────────────────────────────────────────────────────
        encoded_states = np.array([e[0] for e in examples], dtype=np.float32)
        policies = np.array([e[1] for e in examples], dtype=np.float32)
        outcomes = np.array([e[2] for e in examples], dtype=np.float32)

        iter_losses = []
        for _ in range(args["num_epochs"]):
            batch_size = min(args["batch_size"], len(examples))
            idx = np.random.choice(len(examples), size=batch_size, replace=False)
            batch = (encoded_states[idx], policies[idx], outcomes[idx])
            loss = train_step(model, optimizer, batch)
            iter_losses.append(loss)

        avg_loss = sum(iter_losses) / len(iter_losses)
        loss_history.append(avg_loss)

        elapsed = (datetime.now() - iter_start).total_seconds()
        iters_remaining = args["num_iterations"] - iteration - 1
        eta = iters_remaining * elapsed
        print(
            f"Iter {iteration + 1}/{args['num_iterations']} | "
            f"examples={len(examples)} | loss={avg_loss:.4f} | "
            f"elapsed={elapsed:.1f}s | ETA={eta:.0f}s"
        )

        # ── Checkpoint ────────────────────────────────────────────────────
        if (iteration + 1) % args["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"chess_iter_{iteration + 1:04d}.pt"
            )
            save_checkpoint(ckpt_path, iteration + 1, model, optimizer, loss_history, args)

            loss_path = os.path.join(checkpoint_dir, "loss_history.json")
            with open(loss_path, "w") as f:
                json.dump(loss_history, f)

            print(f"  Saved checkpoint: {ckpt_path}")

    return model, loss_history


if __name__ == "__main__":
    run_training()
