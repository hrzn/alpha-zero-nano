"""Re-run Python MCTS against the fixture's lookup table (used as a mock
model) and verify the same visit-count distribution comes back. Guards
against drift between ``mcts/mcts.py`` and the JSON the Rust port consumes.

If this test passes but the Rust parity test fails, the bug is in Rust.
If this test fails, regenerate the fixture via
    uv run python tools/gen_mcts_parity_fixtures.py
"""

import json
import os

import chess
import numpy as np
import pytest

from chess_game.chess_game import ChessGame
from mcts.mcts import MCTS

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "rust", "tests", "fixtures", "mcts_parity.json"
)


@pytest.fixture(scope="module")
def fixture_data():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


class _FixtureModel:
    """Mock model whose `predict` and forward read from the recorded lookup
    table. Exercises the same MCTS code paths Rust will, with the network
    swapped out — so we know the table is sufficient to drive the algorithm
    end-to-end."""

    def __init__(self, lookup: dict, action_size: int = 4096):
        self.lookup = lookup
        self.action_size = action_size

    def _lookup(self, state, player):
        key = f"{state.fen()}|{int(player)}"
        entry = self.lookup.get(key)
        if entry is None:
            raise KeyError(f"missing fixture entry: {key}")
        policy = np.zeros(self.action_size, dtype=np.float32)
        for a, p in zip(entry["policy_actions"], entry["policy_priors"]):
            policy[a] = p
        return policy, float(entry["value"])

    def predict(self, state, player: int):
        return self._lookup(state, player)


def _rerun_mcts(case: dict) -> np.ndarray:
    game = ChessGame()
    model = _FixtureModel(case["lookup"])

    # Rebuild the root by replaying moves (or from initial position).
    board = chess.Board()
    if case["moves_to_reach"]:
        for uci in case["moves_to_reach"]:
            board.push_uci(uci)
    assert board.fen() == case["root_fen"], (
        f"root FEN drift for {case['label']}: built {board.fen()} vs stored {case['root_fen']}"
    )

    # MCTS without Dirichlet, batch_size honoured. Sequential and batched
    # both call _evaluate / _evaluate_batch which we backed by the lookup.
    mcts = MCTS(
        game,
        model=None,  # we'll swap _evaluate / _evaluate_batch below
        num_searches=case["num_searches"],
        c_puct=case["c_puct"],
        batch_size=case["batch_size"],
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )
    mcts._root = None

    def fixture_evaluate(state, p):
        return model._lookup(state, p)

    def fixture_evaluate_batch(leaves):
        return [model._lookup(s, p) for (s, p) in leaves]

    mcts._evaluate = fixture_evaluate
    mcts._evaluate_batch = fixture_evaluate_batch

    return mcts.search(board, case["player"])


def test_each_case_reproduces_visit_policy(fixture_data):
    failures = []
    for case in fixture_data["test_cases"]:
        visit_policy = _rerun_mcts(case)
        # Compare against the stored sparse policy.
        expected = np.zeros(fixture_data["action_size"], dtype=np.float64)
        for a, p in zip(
            case["expected_visit_policy"]["actions"],
            case["expected_visit_policy"]["probs"],
        ):
            expected[a] = p

        diff = float(np.max(np.abs(visit_policy - expected)))
        if diff > 0.0:
            failures.append(
                f"{case['label']}: max abs diff {diff:.6f} (expected 0.0 — fixture self-roundtrip)"
            )
        got_top = int(np.argmax(visit_policy))
        if got_top != case["expected_top_action"]:
            failures.append(
                f"{case['label']}: top action {got_top} vs expected {case['expected_top_action']}"
            )
    assert not failures, "fixture self-test failures:\n  " + "\n  ".join(failures)


def test_lookup_keys_well_formed(fixture_data):
    for case in fixture_data["test_cases"]:
        for key, entry in case["lookup"].items():
            assert "|" in key
            assert len(entry["policy_actions"]) == len(entry["policy_priors"])
            assert all(0 <= a < fixture_data["action_size"] for a in entry["policy_actions"])
            assert all(p >= 0.0 for p in entry["policy_priors"])


def test_schema_header(fixture_data):
    assert fixture_data["schema_version"] == 1
    assert fixture_data["action_size"] == 4096
    assert "model" in fixture_data
    assert len(fixture_data["test_cases"]) > 0
