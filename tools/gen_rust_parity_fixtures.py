"""Emit chess parity fixtures for the Rust port.

For each hand-picked or random chess position, record:
  - FEN (and optional move list for repetition cases)
  - the 17×8×8 encoded tensor (flattened, 1088 floats)
  - sorted list of legal action indices in the player's spatial frame
  - (value, terminated) from get_value_and_terminated
  - each individual is_* predicate so Rust failures localize
  - for promotion-eligible buckets, the explicit (uci, action) list Rust must
    decode to queen-promotion moves

Buckets B1..B11 (see design/RUST_PORT_PLAN.md and /Users/julien/.claude/plans/
purrfect-honking-hammock.md) — each tags one failure mode. Generator exits
non-zero if any bucket ends up empty.

Usage:
    uv run python tools/gen_rust_parity_fixtures.py
    uv run python tools/gen_rust_parity_fixtures.py --out path/to/fixtures.json
"""

import argparse
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import chess
import numpy as np

from chess_game.chess_game import ChessGame

SCHEMA_VERSION = 1
DEFAULT_OUT = "rust/tests/fixtures/chess_parity.json"


@dataclass
class Sample:
    bucket: str
    label: str
    fen: str
    player: int
    encoded: list  # 1088 floats
    legal_actions: list
    value: float
    terminated: bool
    is_checkmate: bool
    is_stalemate: bool
    is_insufficient_material: bool
    is_seventyfive_moves: bool
    is_fivefold_repetition: bool
    moves: Optional[list] = None
    promotion_actions: Optional[list] = None


def _build_sample(game: ChessGame, board: chess.Board, bucket: str, label: str,
                  moves: Optional[list] = None,
                  promotion_actions: Optional[list] = None) -> Sample:
    """Run the live ChessGame against `board` and snapshot every output."""
    player = 1 if board.turn == chess.WHITE else -1
    encoded = game.encode_state(board, player)
    assert encoded.shape == (17, 8, 8) and encoded.dtype == np.float32
    legal_mask = game.get_valid_moves(board)
    legal_actions = sorted(int(a) for a in np.flatnonzero(legal_mask))
    value, terminated = game.get_value_and_terminated(board, action=None)
    return Sample(
        bucket=bucket,
        label=label,
        fen=board.fen(),
        moves=moves,
        player=player,
        encoded=[float(x) for x in encoded.flatten().tolist()],
        legal_actions=legal_actions,
        value=float(value),
        terminated=bool(terminated),
        is_checkmate=bool(board.is_checkmate()),
        is_stalemate=bool(board.is_stalemate()),
        is_insufficient_material=bool(board.is_insufficient_material()),
        is_seventyfive_moves=bool(board.is_seventyfive_moves()),
        is_fivefold_repetition=bool(board.is_fivefold_repetition()),
        promotion_actions=promotion_actions,
    )


def _board_from_moves(start_fen: Optional[str], ucis: list) -> chess.Board:
    """Build a board by replaying moves so history-dependent predicates
    (5-fold repetition, halfmove clock) populate correctly."""
    board = chess.Board(start_fen) if start_fen else chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board


# ── Bucket builders ───────────────────────────────────────────────────────────


def bucket_b1(game) -> list:
    """B1: initial position."""
    return [_build_sample(game, chess.Board(), "B1", "initial_position")]


def bucket_b2(game, rng: np.random.Generator, n: int = 30) -> list:
    """B2: random midgame positions, mix of white-to-move and black-to-move."""
    samples = []
    for i in range(n):
        board = chess.Board()
        n_plies = int(rng.integers(2, 40))
        for _ in range(n_plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[int(rng.integers(0, len(legal)))]
            board.push(move)
            if board.is_game_over(claim_draw=False):
                break
        samples.append(_build_sample(game, board, "B2", f"random_midgame_{i:02d}"))
    return samples


def bucket_b3(game) -> list:
    """B3: checkmate positions — Fool's, Scholar's, smothered."""
    return [
        _build_sample(
            game,
            _board_from_moves(None, ["f2f3", "e7e5", "g2g4", "d8h4"]),
            "B3", "fools_mate",
            moves=["f2f3", "e7e5", "g2g4", "d8h4"],
        ),
        _build_sample(
            game,
            # Scholar's mate: 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#
            _board_from_moves(None, [
                "e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"
            ]),
            "B3", "scholars_mate",
            moves=["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"],
        ),
        _build_sample(
            game,
            # A smothered-mate-style position (constructed directly via FEN).
            chess.Board("6rk/5Npp/8/8/8/8/8/7K b - - 0 1"),
            "B3", "smothered_mate_like",
        ),
    ]


def bucket_b4(game) -> list:
    """B4: stalemate."""
    return [
        _build_sample(
            game,
            chess.Board("k7/8/1QK5/8/8/8/8/8 b - - 0 1"),
            "B4", "stalemate_corner_queen",
        ),
        _build_sample(
            game,
            # Classic stalemate: black king h1, white king f2, white queen g3 — black to move.
            chess.Board("8/8/8/8/8/6Q1/5K2/7k b - - 0 1"),
            "B4", "stalemate_king_queen",
        ),
    ]


def bucket_b5(game) -> list:
    """B5: insufficient material."""
    return [
        _build_sample(
            game,
            chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1"),
            "B5", "king_vs_king",
        ),
        _build_sample(
            game,
            chess.Board("8/8/8/4k3/8/4B3/4K3/8 w - - 0 1"),
            "B5", "king_bishop_vs_king",
        ),
    ]


def bucket_b6(game) -> list:
    """B6: 75-move rule (halfmove clock reaches 150)."""
    # halfmove clock = 150 in FEN field 5
    return [
        _build_sample(
            game,
            chess.Board("8/8/8/4k3/8/8/4K3/R7 w - - 150 80"),
            "B6", "seventyfive_moves",
        ),
    ]


def bucket_b7(game) -> list:
    """B7: 5-fold repetition. Knight shuffle to bring the position back 4 more times."""
    # Each 4-ply cycle (Nc3 Nc6 Nb1 Nb8) returns to starting position.
    # Starting position seen at t=0; need 4 more occurrences → 4 cycles = 16 plies.
    cycle = ["b1c3", "b8c6", "c3b1", "c6b8"]
    moves = cycle * 4
    board = _board_from_moves(None, moves)
    assert board.is_fivefold_repetition(), \
        "expected fivefold repetition; got " + board.fen()
    return [_build_sample(game, board, "B7", "fivefold_knight_shuffle", moves=moves)]


def bucket_b8(game) -> list:
    """B8: every KQkq subset on the same template, to pin castling-plane parity."""
    # Template: kings on e-file, rooks on a/h-files, 8 pawns per side, no
    # other pieces. Valid material counts (shakmaty rejects FENs with too
    # much material on a side).
    base = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w {cr} - 0 1"
    samples = []
    flags = ["K", "Q", "k", "q"]
    for mask in range(16):
        cr = "".join(f for i, f in enumerate(flags) if mask & (1 << i)) or "-"
        fen = base.format(cr=cr)
        samples.append(_build_sample(
            game, chess.Board(fen), "B8", f"castling_{cr if cr != '-' else 'none'}",
        ))
    return samples


def bucket_b9(game) -> list:
    """B9: en passant available — must appear in the legal mask."""
    samples = []
    # White's e5 can capture f6 e.p. after 1.e4 d5 2.e5 f5
    samples.append(_build_sample(
        game,
        _board_from_moves(None, ["e2e4", "d7d5", "e4e5", "f7f5"]),
        "B9", "ep_white_e5xf6",
        moves=["e2e4", "d7d5", "e4e5", "f7f5"],
    ))
    # White's d5 can capture e6 e.p.
    samples.append(_build_sample(
        game,
        _board_from_moves(None, ["d2d4", "g8f6", "d4d5", "e7e5"]),
        "B9", "ep_white_d5xe6",
        moves=["d2d4", "g8f6", "d4d5", "e7e5"],
    ))
    # Black's d4 can capture e3 e.p. after 1.Nf3 d5 2.Ng1 d4 3.e2e4
    samples.append(_build_sample(
        game,
        _board_from_moves(None, ["g1f3", "d7d5", "f3g1", "d5d4", "e2e4"]),
        "B9", "ep_black_d4xe3",
        moves=["g1f3", "d7d5", "f3g1", "d5d4", "e2e4"],
    ))
    # Black's c4 can capture d3 e.p.
    samples.append(_build_sample(
        game,
        _board_from_moves(None, ["b1c3", "c7c5", "c3b1", "c5c4", "d2d4"]),
        "B9", "ep_black_c4xd3",
        moves=["b1c3", "c7c5", "c3b1", "c5c4", "d2d4"],
    ))
    return samples


def bucket_b10(game) -> list:
    """B10: promotion-eligible pawns — Rust must decode to queen promotions."""
    samples = []
    # White pawn on a7, white to move. Legal: a7-a8=Q (and underpromotions, collapsed).
    board = chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    proms = []
    for move in board.legal_moves:
        if (board.piece_at(move.from_square) is not None
                and board.piece_at(move.from_square).piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in (0, 7)
                and move.promotion == chess.QUEEN):
            uci_no_prom = chess.Move(move.from_square, move.to_square).uci()
            action = game.uci_to_action(uci_no_prom, player=1)
            proms.append({"uci": uci_no_prom, "action": int(action)})
    samples.append(_build_sample(
        game, board, "B10", "white_pawn_a7_promote", promotion_actions=proms,
    ))

    # White pawn on h7, multiple capture-promotions available.
    board = chess.Board("6n1/7P/8/8/8/8/8/4K2k w - - 0 1")
    proms = []
    for move in board.legal_moves:
        if (board.piece_at(move.from_square) is not None
                and board.piece_at(move.from_square).piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in (0, 7)
                and move.promotion == chess.QUEEN):
            uci_no_prom = chess.Move(move.from_square, move.to_square).uci()
            action = game.uci_to_action(uci_no_prom, player=1)
            proms.append({"uci": uci_no_prom, "action": int(action)})
    samples.append(_build_sample(
        game, board, "B10", "white_pawn_h7_capture_promote", promotion_actions=proms,
    ))

    # Black pawn on a2, black to move.
    board = chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1")
    proms = []
    for move in board.legal_moves:
        if (board.piece_at(move.from_square) is not None
                and board.piece_at(move.from_square).piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in (0, 7)
                and move.promotion == chess.QUEEN):
            uci_no_prom = chess.Move(move.from_square, move.to_square).uci()
            action = game.uci_to_action(uci_no_prom, player=-1)
            proms.append({"uci": uci_no_prom, "action": int(action)})
    samples.append(_build_sample(
        game, board, "B10", "black_pawn_a2_promote", promotion_actions=proms,
    ))

    # Black pawn on h2 with capture-promote target.
    board = chess.Board("4k3/8/8/8/8/8/7p/4K1N1 b - - 0 1")
    proms = []
    for move in board.legal_moves:
        if (board.piece_at(move.from_square) is not None
                and board.piece_at(move.from_square).piece_type == chess.PAWN
                and chess.square_rank(move.to_square) in (0, 7)
                and move.promotion == chess.QUEEN):
            uci_no_prom = chess.Move(move.from_square, move.to_square).uci()
            action = game.uci_to_action(uci_no_prom, player=-1)
            proms.append({"uci": uci_no_prom, "action": int(action)})
    samples.append(_build_sample(
        game, board, "B10", "black_pawn_h2_capture_promote", promotion_actions=proms,
    ))
    return samples


def bucket_b11(game, rng: np.random.Generator, n: int = 4) -> list:
    """B11: extra black-to-move midgames — the highest-leverage flip-bug catchers."""
    samples = []
    i = 0
    attempts = 0
    while i < n and attempts < 10 * n:
        attempts += 1
        board = chess.Board()
        n_plies = int(rng.integers(1, 25)) * 2 + 1  # odd → black to move
        for _ in range(n_plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            move = legal[int(rng.integers(0, len(legal)))]
            board.push(move)
            if board.is_game_over(claim_draw=False):
                break
        if board.turn == chess.BLACK and not board.is_game_over(claim_draw=False):
            samples.append(_build_sample(game, board, "B11", f"black_to_move_{i:02d}"))
            i += 1
    return samples


# ── Driver ────────────────────────────────────────────────────────────────────


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


def build_all_samples(seed: int = 0) -> list:
    game = ChessGame()
    rng = np.random.default_rng(seed)
    samples = []
    samples += bucket_b1(game)
    samples += bucket_b2(game, rng, n=30)
    samples += bucket_b3(game)
    samples += bucket_b4(game)
    samples += bucket_b5(game)
    samples += bucket_b6(game)
    samples += bucket_b7(game)
    samples += bucket_b8(game)
    samples += bucket_b9(game)
    samples += bucket_b10(game)
    samples += bucket_b11(game, rng, n=4)
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    samples = build_all_samples(seed=args.seed)

    # Ensure every bucket has at least one sample.
    seen = {s.bucket for s in samples}
    expected = {f"B{i}" for i in range(1, 12)}
    missing = expected - seen
    if missing:
        raise SystemExit(f"Missing buckets: {sorted(missing)}")

    out = {
        "schema_version": SCHEMA_VERSION,
        "python_chess_version": chess.__version__,
        "chess_game_source_sha": _git_rev(),
        "num_channels": 17,
        "action_size": 4096,
        "board_shape": [8, 8],
        "samples": [s.__dict__ for s in samples],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)

    print(f"Wrote {args.out}  ({len(samples)} samples across {len(seen)} buckets)")
    by_bucket = {}
    for s in samples:
        by_bucket.setdefault(s.bucket, 0)
        by_bucket[s.bucket] += 1
    for b in sorted(by_bucket):
        print(f"  {b}: {by_bucket[b]}")


if __name__ == "__main__":
    main()
