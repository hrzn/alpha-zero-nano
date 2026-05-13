"""Generate reference (state, logits, value) tuples by running random Connect4
positions through the trained PyTorch model. The TypeScript port consumes
the resulting JSON to verify it produces matching outputs.

Usage:
    uv run python tools/gen_agreement_data.py
    uv run python tools/gen_agreement_data.py --n 100 --out web/tools/agreement_ref.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from connect4 import Connect4
from model.model import ResNet

DEFAULT_CKPT_GLOB = "checkpoints/c4/connect4_iter_*.pt"
DEFAULT_OUT = "web/tools/agreement_ref.json"


def random_position(game: Connect4, rng: np.random.Generator, max_moves: int) -> tuple[np.ndarray, int]:
    """Play `max_moves` random moves; return resulting state and the player to move next."""
    state = game.get_initial_state()
    player = 1
    for _ in range(max_moves):
        valid = game.get_valid_moves(state)
        if not valid.any():
            break
        legal = np.flatnonzero(valid)
        action = int(rng.choice(legal))
        state = game.update_state(state, action, player)
        _, terminated = game.get_value_and_terminated(state, action)
        if terminated:
            break
        player = game.get_opponent(player)
    return state, player


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help=f"Checkpoint .pt (default: latest matching {DEFAULT_CKPT_GLOB})")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--n", type=int, default=50, help="Number of reference samples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.ckpt is None:
        files = sorted(glob.glob(DEFAULT_CKPT_GLOB))
        if not files:
            raise FileNotFoundError(f"No checkpoint matching {DEFAULT_CKPT_GLOB}")
        args.ckpt = files[-1]

    print(f"Loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    game = Connect4()
    model_args = ckpt.get("args", {})
    model = ResNet(
        game,
        num_res_blocks=model_args.get("num_res_blocks", 3),
        num_hidden=model_args.get("num_hidden", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    samples = []
    # Spread across early/mid/late game positions.
    for i in range(args.n):
        n_moves = int(rng.integers(0, 30))
        state, player = random_position(game, rng, n_moves)
        encoded = game.encode_state(state, player)
        with torch.no_grad():
            x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
            policy_logits, value = model(x)
            logits = policy_logits.squeeze(0).cpu().numpy().tolist()
            v = float(value.item())
        samples.append(
            {
                "state": state.astype(int).flatten().tolist(),  # 42 ints
                "player": int(player),
                "n_moves_played": n_moves,
                "policy_logits": logits,
                "value": v,
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "source_checkpoint": os.path.basename(args.ckpt),
                "model_args": {
                    "num_res_blocks": model_args.get("num_res_blocks", 3),
                    "num_hidden": model_args.get("num_hidden", 64),
                },
                "samples": samples,
            },
            f,
        )
    print(f"Wrote {args.out}  ({len(samples)} samples)")


if __name__ == "__main__":
    main()
