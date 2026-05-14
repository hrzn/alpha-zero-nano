"""Round-trip every fixture in rust/tests/fixtures/chess_parity.json against
the live ChessGame to guarantee the JSON cannot silently drift from
chess_game.py. Run as part of the regular Python test suite.

The Rust port reads the same file and asserts the same equalities. If this
test passes but the Rust parity test fails, the bug is in Rust. If this test
fails, the fixture is stale — regenerate via
    uv run python tools/gen_rust_parity_fixtures.py
"""

import json
import os

import chess
import numpy as np
import pytest

from chess_game.chess_game import ChessGame

FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "rust", "tests", "fixtures", "chess_parity.json"
)
EXPECTED_BUCKETS = {f"B{i}" for i in range(1, 12)}


@pytest.fixture(scope="module")
def fixture_data():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def game():
    return ChessGame()


def _build_state(sample):
    """Rebuild the board exactly as the generator did: FEN, optionally
    replaying UCI moves so history-dependent predicates populate."""
    if sample["moves"]:
        board = chess.Board()
        for uci in sample["moves"]:
            board.push_uci(uci)
    else:
        board = chess.Board(sample["fen"])
    return board


def test_schema_header(fixture_data):
    assert fixture_data["schema_version"] == 1
    assert fixture_data["num_channels"] == 17
    assert fixture_data["action_size"] == 4096
    assert fixture_data["board_shape"] == [8, 8]
    assert len(fixture_data["samples"]) > 0


def test_all_buckets_present(fixture_data):
    seen = {s["bucket"] for s in fixture_data["samples"]}
    assert seen == EXPECTED_BUCKETS, f"missing buckets: {EXPECTED_BUCKETS - seen}"


def test_each_sample_roundtrips(fixture_data, game):
    """Encoder, legal actions, value/terminated, and every is_* predicate
    must match what the live ChessGame produces from the stored FEN/moves."""
    failures = []
    for i, sample in enumerate(fixture_data["samples"]):
        board = _build_state(sample)
        tag = f"sample[{i}] bucket={sample['bucket']} label={sample['label']}"

        # FEN consistency (moves-replayed boards should produce the stored FEN)
        if board.fen() != sample["fen"]:
            failures.append(f"{tag}: fen mismatch {board.fen()} vs {sample['fen']}")
            continue

        # Encoder
        encoded = game.encode_state(board, sample["player"]).flatten()
        if not np.array_equal(encoded, np.asarray(sample["encoded"], dtype=np.float32)):
            failures.append(f"{tag}: encoded tensor mismatch")
            continue

        # Legal actions
        mask = game.get_valid_moves(board)
        actions = sorted(int(a) for a in np.flatnonzero(mask))
        if actions != sample["legal_actions"]:
            failures.append(f"{tag}: legal_actions mismatch")
            continue

        # Termination + per-predicate
        v, term = game.get_value_and_terminated(board, action=None)
        if float(v) != sample["value"] or bool(term) != sample["terminated"]:
            failures.append(f"{tag}: (value, terminated) mismatch")
            continue
        for name, expected in [
            ("is_checkmate", sample["is_checkmate"]),
            ("is_stalemate", sample["is_stalemate"]),
            ("is_insufficient_material", sample["is_insufficient_material"]),
            ("is_seventyfive_moves", sample["is_seventyfive_moves"]),
            ("is_fivefold_repetition", sample["is_fivefold_repetition"]),
        ]:
            if bool(getattr(board, name)()) != expected:
                failures.append(f"{tag}: {name} mismatch")
                break

        # Promotion actions, where present, must materialise as queen promotions.
        if sample.get("promotion_actions"):
            for entry in sample["promotion_actions"]:
                action = entry["action"]
                # Decode via _action_to_move (private but the parity contract).
                move = game._action_to_move(board, action)
                if move.promotion != chess.QUEEN:
                    failures.append(f"{tag}: action {action} not queen promotion")

    assert not failures, "fixture-vs-live mismatches:\n  " + "\n  ".join(failures)
