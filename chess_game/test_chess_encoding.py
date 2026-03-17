"""Tests for the Chess board encoding."""

import chess
import numpy as np
import pytest

from chess_game.chess_game import ChessGame


@pytest.fixture
def game():
    return ChessGame()


class TestChessEncodingShape:
    def test_num_channels(self, game):
        assert game.num_channels == 17

    def test_encoded_shape(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        assert enc.shape == (17, 8, 8)

    def test_encoded_dtype(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        assert enc.dtype == np.float32


class TestChessEncodingPiecePlanes:
    def test_white_pawns_in_channel_0_for_white(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        # White pawns start on rank 2 (index 1 from white's perspective)
        assert enc[0, 1, :].sum() == 8  # 8 white pawns

    def test_black_pawns_in_opponent_channel_for_white(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        # Opponent (black) pawns in channels 6-11; black pawns -> channel 6
        assert enc[6, 6, :].sum() == 8  # 8 black pawns (rank 7, index 6 from white's view)

    def test_board_flipped_for_black(self, game):
        state = game.get_initial_state()
        enc_white = game.encode_state(state, player=1)
        enc_black = game.encode_state(state, player=-1)

        # From black's perspective, black's own pawns (channel 0) are on row index 1
        assert enc_black[0, 1, :].sum() == 8

    def test_empty_squares_have_no_pieces(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        # Ranks 3-6 (indices 2-5) are empty in the initial position
        assert enc[:12, 2:6, :].sum() == 0


class TestChessEncodingCastling:
    def test_all_castling_rights_set_at_start(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        # Channels 12-15: all castling rights available at game start
        assert enc[12].sum() == 64  # current player kingside (all 1s)
        assert enc[13].sum() == 64  # current player queenside
        assert enc[14].sum() == 64  # opponent kingside
        assert enc[15].sum() == 64  # opponent queenside

    def test_no_castling_rights_after_loss(self, game):
        # Position with no castling rights
        board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
        enc = game.encode_state(board, player=1)
        assert enc[12].sum() == 0
        assert enc[13].sum() == 0


class TestChessEncodingColorPlane:
    def test_color_plane_is_1_for_white(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=1)
        assert enc[16].sum() == 64  # all ones

    def test_color_plane_is_0_for_black(self, game):
        state = game.get_initial_state()
        enc = game.encode_state(state, player=-1)
        assert enc[16].sum() == 0  # all zeros
