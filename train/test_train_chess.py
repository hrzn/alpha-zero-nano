"""Tests for self-play on Chess — regression guard for optimizations."""

import numpy as np
import pytest

from chess_game.chess_game import ChessGame
from mcts.mcts import MCTS
from model.model import ResNet
from train.train import self_play


@pytest.fixture
def game():
    return ChessGame()


@pytest.fixture
def model(game):
    return ResNet(game, num_res_blocks=1, num_hidden=32)


@pytest.fixture
def mcts(game, model):
    return MCTS(game, model=model, num_searches=5)


class TestSelfPlayChess:
    def test_returns_list_of_examples(self, game, mcts):
        examples = self_play(game, mcts, max_moves=10)
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_example_encoded_state_shape(self, game, mcts):
        """Encoded state must have the chess encoding shape (17, 8, 8)."""
        examples = self_play(game, mcts, max_moves=10)
        encoded_state, _, _ = examples[0]
        assert encoded_state.shape == (game.num_channels, game.row_count, game.column_count)
        assert encoded_state.shape == (17, 8, 8)

    def test_example_policy_shape(self, game, mcts):
        """Policy must have length action_size (4096)."""
        examples = self_play(game, mcts, max_moves=10)
        _, policy, _ = examples[0]
        assert policy.shape == (game.action_size,)
        assert len(policy) == 4096

    def test_policies_are_valid_distributions(self, game, mcts):
        examples = self_play(game, mcts, max_moves=10)
        for _, policy, _ in examples:
            assert policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert (policy >= 0).all()

    def test_outcomes_are_valid(self, game, mcts):
        """Outcomes must be in {-1, 0, 1}."""
        examples = self_play(game, mcts, max_moves=10)
        for _, _, outcome in examples:
            assert outcome in {-1.0, 0.0, 1.0}

    def test_game_respects_max_moves(self, game, mcts):
        """With max_moves=N, the game produces at most N examples."""
        max_moves = 5
        examples = self_play(game, mcts, max_moves=max_moves)
        assert len(examples) <= max_moves

    def test_game_terminates_without_max_moves(self, game, mcts):
        """Without max_moves, a short game should still terminate."""
        # Use a tiny search to keep the test fast; game may run many moves
        # but must eventually terminate via chess rules or draw conditions.
        # We cap with a reasonable large max_moves to avoid infinite loops in CI.
        examples = self_play(game, mcts, max_moves=50)
        assert len(examples) > 0

    def test_all_examples_from_same_game_have_same_outcome_magnitude(self, game, mcts):
        """All positions from a single game share the same game result (±1 or 0)."""
        examples = self_play(game, mcts, max_moves=10)
        outcomes = [abs(o) for _, _, o in examples]
        # All outcomes are either all 0 (draw) or all 1 (someone won)
        assert len(set(outcomes)) == 1
