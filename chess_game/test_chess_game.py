"""Tests for the Chess game environment."""

import chess
import numpy as np
import pytest

from chess_game.chess_game import ChessGame


@pytest.fixture
def game():
    return ChessGame()


def _apply_moves(game, state, ucis):
    """Apply a sequence of UCI moves, alternating player (white first).
    Returns (final_state, last_action)."""
    player = 1
    action = None
    for uci in ucis:
        action = game.uci_to_action(uci, player=player)
        state = game.update_state(state, action, player=player)
        player = game.get_opponent(player)
    return state, action


class TestChessGameInterface:
    def test_action_size(self, game):
        # 64 from-squares × 64 to-squares
        assert game.action_size == 64 * 64

    def test_board_dimensions(self, game):
        assert game.row_count == 8
        assert game.column_count == 8

    def test_initial_state_is_chess_board(self, game):
        state = game.get_initial_state()
        assert isinstance(state, chess.Board)

    def test_initial_state_is_starting_position(self, game):
        state = game.get_initial_state()
        assert state.fen().startswith(chess.STARTING_FEN.split(' ')[0])

    def test_get_opponent(self, game):
        assert game.get_opponent(1) == -1
        assert game.get_opponent(-1) == 1


class TestValidMovesChess:
    def test_initial_position_has_20_legal_moves(self, game):
        state = game.get_initial_state()
        valid = game.get_valid_moves(state)
        assert valid.sum() == 20

    def test_valid_moves_length(self, game):
        state = game.get_initial_state()
        valid = game.get_valid_moves(state)
        assert len(valid) == game.action_size

    def test_valid_moves_are_binary(self, game):
        state = game.get_initial_state()
        valid = game.get_valid_moves(state)
        assert set(valid).issubset({0, 1})

    def test_no_valid_moves_in_checkmate(self, game):
        # Fool's mate: 1.f3 e5 2.g4 Qh4#
        state = game.get_initial_state()
        state, _ = _apply_moves(game, state, ["f2f3", "e7e5", "g2g4", "d8h4"])
        valid = game.get_valid_moves(state)
        assert valid.sum() == 0


class TestUpdateState:
    def test_update_state_does_not_mutate_original(self, game):
        state = game.get_initial_state()
        original_fen = state.fen()
        action = game.uci_to_action("e2e4")
        game.update_state(state, action, player=1)
        assert state.fen() == original_fen

    def test_update_state_applies_move(self, game):
        state = game.get_initial_state()
        action = game.uci_to_action("e2e4")
        new_state = game.update_state(state, action, player=1)
        # e2 should be empty, e4 should have a white pawn
        assert new_state.piece_at(chess.E4) is not None
        assert new_state.piece_at(chess.E2) is None

    def test_action_to_uci_roundtrip(self, game):
        # White moves
        for uci in ["e2e4", "g1f3"]:
            action = game.uci_to_action(uci, player=1)
            assert 0 <= action < game.action_size
            assert game.action_to_uci(action, player=1) == uci
        # Black moves
        for uci in ["d7d5", "e7e5"]:
            action = game.uci_to_action(uci, player=-1)
            assert 0 <= action < game.action_size
            assert game.action_to_uci(action, player=-1) == uci


class TestTermination:
    def test_ongoing_game_not_terminated(self, game):
        state = game.get_initial_state()
        action = game.uci_to_action("e2e4")
        state = game.update_state(state, action, player=1)
        value, terminated = game.get_value_and_terminated(state, action)
        assert not terminated

    def test_checkmate_is_terminal_with_value_1(self, game):
        # Fool's mate — the last move (Qh4#) is checkmate
        state = game.get_initial_state()
        state, action = _apply_moves(game, state, ["f2f3", "e7e5", "g2g4", "d8h4"])
        value, terminated = game.get_value_and_terminated(state, action)
        assert terminated
        assert value == 1

    def test_stalemate_is_terminal_with_value_0(self, game):
        # Set up a known stalemate position via FEN
        # Black king on a8, white queen on b6, white king on c6 — black to move, stalemate
        board = chess.Board("k7/8/1QK5/8/8/8/8/8 b - - 0 1")
        # No last action needed — just check state
        value, terminated = game.get_value_and_terminated(board, action=None)
        assert terminated
        assert value == 0


class TestActionFlipSymmetry:
    """Verify that actions are in the same spatial frame as encode_state."""

    def test_white_pawn_e2e4_action_matches_encoding(self, game):
        """White's e2→e4: pawn is at encoding row 1, action should use row 1."""
        state = game.get_initial_state()
        action = game.uci_to_action("e2e4", player=1)
        from_sq = action // 64
        from_row, from_col = from_sq // 8, from_sq % 8
        # White's e2 pawn is at rank 1 → encoding row 1
        assert from_row == 1
        assert from_col == 4

    def test_black_e7e5_action_is_flipped(self, game):
        """Black's e7→e5: pawn at rank 6 is at encoding row 1 (flipped)."""
        action = game.uci_to_action("e7e5", player=-1)
        from_sq = action // 64
        from_row = from_sq // 8
        # Black's e7 pawn (rank 6) should be at flipped row 1
        assert from_row == 1

    def test_valid_moves_match_encoding_frame(self, game):
        """After 1.e4, black's valid moves should be in flipped coordinates."""
        state = game.get_initial_state()
        state = game.update_state(state, game.uci_to_action("e2e4", player=1), 1)
        valid = game.get_valid_moves(state)  # black to move
        # e7→e5 in flipped coords: from row 1, col 4 → to row 3, col 4
        flipped_action = game.uci_to_action("e7e5", player=-1)
        assert valid[flipped_action] == 1

    def test_roundtrip_through_update_state(self, game):
        """Action encoded for black should round-trip through update_state."""
        state = game.get_initial_state()
        state = game.update_state(state, game.uci_to_action("e2e4", player=1), 1)
        # Black plays e7e5
        action = game.uci_to_action("e7e5", player=-1)
        new_state = game.update_state(state, action, -1)
        # e7 should be empty, e5 should have a black pawn
        assert new_state.piece_at(chess.E5) is not None
        assert new_state.piece_at(chess.E7) is None
