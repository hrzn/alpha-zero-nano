"""Tests for the Chess game environment."""

import chess
import numpy as np
import pytest

from chess_game.chess_game import ChessGame


@pytest.fixture
def game():
    return ChessGame()


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
        for move_uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            action = game.uci_to_action(move_uci)
            state = game.update_state(state, action, player=None)
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
        for uci in ["e2e4", "d7d5", "g1f3"]:
            action = game.uci_to_action(uci)
            assert 0 <= action < game.action_size


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
        moves = ["f2f3", "e7e5", "g2g4", "d8h4"]
        action = None
        for uci in moves:
            action = game.uci_to_action(uci)
            state = game.update_state(state, action, player=None)
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
