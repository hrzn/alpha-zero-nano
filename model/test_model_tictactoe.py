"""Tests for the ResNet model on Tic-tac-toe."""

import numpy as np
import pytest
import torch

from model.model import ResNet
from tictactoe.tictactoe import TicTacToe


@pytest.fixture
def game():
    return TicTacToe()


@pytest.fixture
def model(game):
    return ResNet(game, num_res_blocks=2, num_hidden=64)


class TestStateEncodingTicTacToe:
    def test_empty_board_encoding(self, game, model):
        state = game.get_initial_state()
        encoded = game.encode_state(state, player=1)

        # Shape: (3, rows, cols)
        assert encoded.shape == (3, game.row_count, game.column_count)
        assert (encoded[0] == 0).all()  # no player 1 pieces
        assert (encoded[1] == 0).all()  # no player 2 pieces
        assert (encoded[2] == 1).all()  # all squares empty

    def test_player1_pieces_in_channel_0(self, game, model):
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)  # player 1 at top-left
        state = game.update_state(state, 4, -1)  # player 2 at center
        encoded = game.encode_state(state, player=1)

        assert encoded[0, 0, 0] == 1  # player 1 at position (0,0)
        assert encoded[0, 1, 1] == 0  # player 1 not at center
        assert encoded[1, 0, 0] == 0  # player 2 not at (0,0)
        assert encoded[1, 1, 1] == 1  # player 2 at center

    def test_encoding_flips_for_player2(self, game, model):
        """From player 2's perspective, channels 0 and 1 should be swapped."""
        state = game.get_initial_state()
        state = game.update_state(state, 0, 1)   # player 1 at top-left
        state = game.update_state(state, 4, -1)  # player 2 at center

        enc1 = game.encode_state(state, player=1)
        enc2 = game.encode_state(state, player=-1)

        # From player 2's view, their pieces are in channel 0
        assert enc2[0, 1, 1] == 1   # player 2's piece in "own" channel
        assert enc2[1, 0, 0] == 1   # player 1's piece in "opponent" channel
        # From player 1's view, their pieces are in channel 0
        assert enc1[0, 0, 0] == 1
        assert enc1[1, 1, 1] == 1


class TestModelOutputsTicTacToe:
    def test_policy_shape(self, game, model):
        state = game.get_initial_state()
        policy, value = model.predict(state, player=1)

        assert policy.shape == (game.action_size,)

    def test_value_is_scalar(self, game, model):
        state = game.get_initial_state()
        policy, value = model.predict(state, player=1)

        assert np.isscalar(value) or value.shape == ()

    def test_policy_sums_to_one(self, game, model):
        state = game.get_initial_state()
        policy, value = model.predict(state, player=1)

        assert policy.sum() == pytest.approx(1.0, abs=1e-5)
        assert (policy >= 0).all()

    def test_value_in_range(self, game, model):
        state = game.get_initial_state()
        policy, value = model.predict(state, player=1)

        assert -1.0 <= float(value) <= 1.0

    def test_different_states_give_different_outputs(self, game, model):
        state1 = game.get_initial_state()
        state2 = game.get_initial_state()
        state2 = game.update_state(state2, 4, 1)  # center occupied

        policy1, _ = model.predict(state1, player=1)
        policy2, _ = model.predict(state2, player=1)

        assert not np.allclose(policy1, policy2)


class TestModelArchitectureTicTacToe:
    def test_small_model(self, game):
        model = ResNet(game, num_res_blocks=1, num_hidden=32)
        state = game.get_initial_state()
        policy, value = model.predict(state, player=1)

        assert policy.shape == (game.action_size,)
        assert -1.0 <= float(value) <= 1.0

    def test_larger_model(self, game):
        model = ResNet(game, num_res_blocks=4, num_hidden=128)
        state = game.get_initial_state()
        policy, value = model.predict(state, player=1)

        assert policy.shape == (game.action_size,)
        assert -1.0 <= float(value) <= 1.0

    def test_backward_pass(self, game, model):
        """Model must be differentiable for training."""
        state = game.get_initial_state()
        encoded = game.encode_state(state, player=1)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)

        policy_logits, value = model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()  # should not raise
