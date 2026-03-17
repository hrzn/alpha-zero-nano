"""Tests for the ResNet model on Chess."""

import numpy as np
import pytest
import torch

from chess_game.chess_game import ChessGame
from model.model import ResNet


@pytest.fixture
def game():
    return ChessGame()


@pytest.fixture
def model(game):
    return ResNet(game, num_res_blocks=2, num_hidden=64)


class TestModelOutputsChess:
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


class TestModelArchitectureChess:
    def test_input_channels_match_game(self, game, model):
        """The model's first conv layer must accept game.num_channels inputs."""
        first_conv = model.input_block[0]
        assert first_conv.in_channels == game.num_channels

    def test_backward_pass(self, game, model):
        state = game.get_initial_state()
        encoded = game.encode_state(state, player=1)
        x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)
        policy_logits, value = model(x)
        loss = policy_logits.sum() + value.sum()
        loss.backward()  # should not raise
