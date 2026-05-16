"""Emit MCTS parity fixtures for the Rust port.

For each (root position, num_searches, c_puct, batch_size) test case:
- Run Python MCTS with deterministic settings (Dirichlet OFF, fixed model
  seed, no randomness anywhere).
- Intercept every model query and record the lookup table
  `(fen, player) -> (policy_priors, value)`. The Rust port consumes this
  table via a `FixtureEvaluator` so the MCTS algorithm itself is what's
  under test, not ONNX inference.
- Record the final visit-count distribution + argmax for the parity test
  to assert against.

Usage:
    uv run python tools/gen_mcts_parity_fixtures.py
    uv run python tools/gen_mcts_parity_fixtures.py --out path/to.json
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np
import torch

from chess_game.chess_game import ChessGame
from mcts.mcts import MCTS
from model.model import ResNet

SCHEMA_VERSION = 1
DEFAULT_OUT = "rust/tests/fixtures/mcts_parity.json"

# Small model shape so fixture generation is fast. Parity is about the
# algorithm, not the network shape — these test cases are just as
# discriminating with a small net as with the S preset.
MODEL_NUM_RES_BLOCKS = 3
MODEL_NUM_HIDDEN = 64
MODEL_SEED = 0


@dataclass
class TestCase:
    label: str
    root_fen: str
    moves_to_reach: Optional[list]  # UCI sequence to build the root from initial
    player: int
    num_searches: int
    c_puct: float
    batch_size: int
    dirichlet_alpha: float
    dirichlet_epsilon: float
    expected_visit_policy: dict  # {"actions": [...], "probs": [...]}, sparse
    expected_top_action: int
    lookup: dict  # "<fen>|<player_int>" -> {"policy": [4096], "value": float}


def _build_model() -> ResNet:
    torch.manual_seed(MODEL_SEED)
    game = ChessGame()
    model = ResNet(game, num_res_blocks=MODEL_NUM_RES_BLOCKS, num_hidden=MODEL_NUM_HIDDEN)
    model.eval()
    return model


def _build_state(moves_to_reach: Optional[list]) -> chess.Board:
    board = chess.Board()
    if moves_to_reach:
        for uci in moves_to_reach:
            board.push_uci(uci)
    return board


def _run_recording_mcts(
    game: ChessGame,
    model: ResNet,
    root_state: chess.Board,
    player: int,
    num_searches: int,
    c_puct: float,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    """Run MCTS with Dirichlet OFF. Return (visit_policy[4096], lookup_table).

    The lookup table records every (fen, player) -> (policy_priors, value)
    the network was queried for during the search.
    """
    mcts = MCTS(
        game,
        model=model,
        num_searches=num_searches,
        c_puct=c_puct,
        batch_size=batch_size,
        dirichlet_alpha=0.0,      # OFF
        dirichlet_epsilon=0.0,    # belt and braces — _apply_dirichlet_noise is a no-op when alpha<=0
    )
    mcts._root = None  # fresh tree

    lookup: dict = {}

    # Monkey-patch _evaluate (sequential leaves) and _evaluate_batch
    # (batched leaves) to record every (fen, player) -> (policy, value).
    orig_evaluate = mcts._evaluate
    orig_evaluate_batch = mcts._evaluate_batch

    def _record(state, p, policy, value):
        # Sparse storage: only priors at legal-action indices. MCTS masks
        # illegal indices to zero before renormalising, so storing them is
        # waste — the algorithm cannot use them. Cuts fixture size ~40×.
        legal_mask = game.get_valid_moves(state)
        actions = [int(a) for a in np.flatnonzero(legal_mask)]
        priors = [float(policy[a]) for a in actions]
        key = f"{state.fen()}|{int(p)}"
        lookup[key] = {
            "policy_actions": actions,
            "policy_priors": priors,
            "value": float(value),
        }

    def recording_evaluate(state, p):
        policy, value = orig_evaluate(state, p)
        _record(state, p, policy, value)
        return policy, value

    def recording_evaluate_batch(leaves):
        results = orig_evaluate_batch(leaves)
        for (state, p), (policy, value) in zip(leaves, results):
            _record(state, p, policy, value)
        return results

    mcts._evaluate = recording_evaluate
    mcts._evaluate_batch = recording_evaluate_batch

    policy = mcts.search(root_state, player)
    return policy, lookup


def _make_case(
    game: ChessGame,
    model: ResNet,
    label: str,
    moves_to_reach: Optional[list],
    num_searches: int,
    c_puct: float = 1.0,
    batch_size: int = 1,
) -> TestCase:
    board = _build_state(moves_to_reach)
    player = 1 if board.turn == chess.WHITE else -1
    visit_policy, lookup = _run_recording_mcts(
        game, model, board, player, num_searches, c_puct, batch_size
    )
    top_action = int(np.argmax(visit_policy))
    # Sparse expected policy too — at most ~30 legal actions at root.
    nonzero = np.flatnonzero(visit_policy)
    return TestCase(
        label=label,
        root_fen=board.fen(),
        moves_to_reach=moves_to_reach,
        player=player,
        num_searches=num_searches,
        c_puct=c_puct,
        batch_size=batch_size,
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
        expected_visit_policy={
            "actions": [int(a) for a in nonzero],
            "probs": [float(visit_policy[a]) for a in nonzero],
        },
        expected_top_action=top_action,
        lookup=lookup,
    )


def build_all_cases() -> list:
    """Construct the suite of MCTS parity test cases."""
    game = ChessGame()
    model = _build_model()
    cases: list = []

    # ── Sequential (batch_size=1) ────────────────────────────────────────
    # TC1: initial position, modest search depth.
    cases.append(_make_case(game, model, "seq_initial_50", None, 50))
    # TC2: initial position, deeper search + different c_puct.
    cases.append(_make_case(game, model, "seq_initial_100_cpuct1.5", None, 100, c_puct=1.5))
    # TC3: mid-game position, white to move.
    cases.append(_make_case(
        game, model, "seq_midgame_white_50",
        ["e2e4", "e7e5", "g1f3", "b8c6"], 50,
    ))
    # TC4: mid-game position, black to move — exercises perspective flip
    # through the whole MCTS algorithm, not just the encoder.
    cases.append(_make_case(
        game, model, "seq_midgame_black_50",
        ["e2e4", "e7e5", "g1f3"], 50,
    ))

    # ── Batched (batch_size > 1, virtual loss) ───────────────────────────
    # TC5: initial position, batch of 8, total 64 sims.
    cases.append(_make_case(
        game, model, "batch_initial_64_bs8", None, 64, batch_size=8,
    ))
    # TC6: mid-game, batched.
    cases.append(_make_case(
        game, model, "batch_midgame_white_64_bs8",
        ["e2e4", "e7e5", "g1f3", "b8c6"], 64, batch_size=8,
    ))

    return cases


def _git_rev() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    cases = build_all_cases()
    out = {
        "schema_version": SCHEMA_VERSION,
        "mcts_source_sha": _git_rev(),
        "action_size": 4096,
        "model": {
            "num_res_blocks": MODEL_NUM_RES_BLOCKS,
            "num_hidden": MODEL_NUM_HIDDEN,
            "seed": MODEL_SEED,
        },
        "test_cases": [c.__dict__ for c in cases],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out}  ({len(cases)} test cases, {size_mb:.1f} MB)")
    for c in cases:
        print(
            f"  {c.label:32s}  sims={c.num_searches:3d}  bs={c.batch_size:2d}  "
            f"lookup={len(c.lookup):3d}  top={c.expected_top_action}"
        )


if __name__ == "__main__":
    main()
