"""Tests for the self-play and training loop on Tic-tac-toe."""

import numpy as np
import pytest
import torch
import torch.optim as optim

from mcts.mcts import MCTS
from model.model import ResNet
from tictactoe.tictactoe import TicTacToe
from train.train import AlphaZero, parallel_self_play, self_play, train_step


@pytest.fixture
def game():
    return TicTacToe()


@pytest.fixture
def model(game):
    return ResNet(game, num_res_blocks=2, num_hidden=64)


@pytest.fixture
def mcts(game, model):
    return MCTS(game, model=model, num_searches=20)


class TestSelfPlayTicTacToe:
    def test_returns_list_of_examples(self, game, mcts):
        examples = self_play(game, mcts)
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_example_structure(self, game, mcts):
        """Each example is a (encoded_state, policy, outcome) tuple."""
        examples = self_play(game, mcts)
        encoded_state, policy, outcome = examples[0]

        assert encoded_state.shape == (3, game.row_count, game.column_count)
        assert policy.shape == (game.action_size,)
        assert isinstance(outcome, float)

    def test_game_terminates(self, game, mcts):
        """A self-play game must terminate — at most action_size moves."""
        examples = self_play(game, mcts)
        assert len(examples) <= game.action_size

    def test_policies_are_valid_distributions(self, game, mcts):
        examples = self_play(game, mcts)
        for _, policy, _ in examples:
            assert policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert (policy >= 0).all()

    def test_outcomes_are_valid(self, game, mcts):
        """Outcomes must be -1, 0, or 1 from each player's perspective."""
        examples = self_play(game, mcts)
        for _, _, outcome in examples:
            assert outcome in {-1.0, 0.0, 1.0}


class TestTrainStepTicTacToe:
    def test_returns_scalar_loss(self, game, model):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch(game, model, batch_size=4)
        loss = train_step(model, optimizer, batch)

        assert isinstance(loss, float)
        assert loss > 0

    def test_parameters_change_after_step(self, game, model):
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        batch = _make_batch(game, model, batch_size=4)

        params_before = [p.clone() for p in model.parameters()]
        train_step(model, optimizer, batch)
        params_after = list(model.parameters())

        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )

    def test_loss_decreases_over_many_steps(self, game, model):
        """Loss should trend downward when repeatedly training on the same batch."""
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        batch = _make_batch(game, model, batch_size=16)

        losses = [train_step(model, optimizer, batch) for _ in range(30)]

        assert losses[-1] < losses[0]


@pytest.fixture
def args():
    return {
        "num_searches": 10,
        "num_self_play_games": 2,
        "num_epochs": 2,
        "batch_size": 8,
        "lr": 1e-3,
    }


class TestAlphaZeroTicTacToe:
    def test_run_one_iteration_completes(self, game, model, args):
        az = AlphaZero(game, model, args)
        az.run(num_iterations=1)  # should not raise

    def test_model_parameters_change_after_run(self, game, model, args):
        params_before = [p.clone() for p in model.parameters()]
        az = AlphaZero(game, model, args)
        az.run(num_iterations=1)
        params_after = list(model.parameters())

        assert any(
            not torch.equal(before, after)
            for before, after in zip(params_before, params_after)
        )

    def test_examples_accumulate_across_games(self, game, model, args):
        """Multiple self-play games should yield more examples than a single game."""
        mcts = MCTS(game, model=model, num_searches=args["num_searches"])
        single_game = self_play(game, mcts)

        examples_4_games = []
        for _ in range(4):
            examples_4_games += self_play(game, mcts)
        assert len(examples_4_games) >= len(single_game)

    def test_batch_size_larger_than_examples(self, game, model):
        """Training should work even when batch_size > number of examples."""
        args = {
            "num_searches": 5,
            "num_self_play_games": 1,
            "num_epochs": 1,
            "batch_size": 512,  # much larger than examples from 1 game
            "lr": 1e-3,
        }
        az = AlphaZero(game, model, args)
        az.run(num_iterations=1)  # should not raise

    def test_checkpoint_save_and_load(self, game, model, args, tmp_path):
        az = AlphaZero(game, model, args)
        az.run(num_iterations=1)

        checkpoint_path = tmp_path / "model.pt"
        az.save(checkpoint_path)

        new_model = ResNet(game, num_res_blocks=2, num_hidden=64)
        az2 = AlphaZero(game, new_model, args)
        az2.load(checkpoint_path)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)


class TestParallelSelfPlay:
    """Opt 3: parallel_self_play runs multiple games via multiprocessing."""

    def test_returns_correct_number_of_games_single_worker(self, game, model):
        """n_workers=1 (no multiprocessing) should return examples from n_games games."""
        mcts = MCTS(game, model=model, num_searches=5)
        examples = parallel_self_play(game, mcts, n_games=3, max_moves=None, n_workers=1)
        assert isinstance(examples, list)
        assert len(examples) >= 3  # at least 1 example per game

    def test_examples_have_correct_shapes_single_worker(self, game, model):
        mcts = MCTS(game, model=model, num_searches=5)
        examples = parallel_self_play(game, mcts, n_games=2, max_moves=None, n_workers=1)
        for enc_state, policy, outcome in examples:
            assert enc_state.shape == (3, game.row_count, game.column_count)
            assert policy.shape == (game.action_size,)
            assert isinstance(outcome, float)

    def test_policies_valid_single_worker(self, game, model):
        mcts = MCTS(game, model=model, num_searches=5)
        examples = parallel_self_play(game, mcts, n_games=2, max_moves=None, n_workers=1)
        for _, policy, _ in examples:
            assert policy.sum() == pytest.approx(1.0, abs=1e-5)
            assert (policy >= 0).all()

    def test_outcomes_valid_single_worker(self, game, model):
        mcts = MCTS(game, model=model, num_searches=5)
        examples = parallel_self_play(game, mcts, n_games=2, max_moves=None, n_workers=1)
        for _, _, outcome in examples:
            assert outcome in {-1.0, 0.0, 1.0}

    def test_multi_worker_returns_examples(self, game, model):
        """Multi-worker path should also return a valid list of examples."""
        mcts = MCTS(game, model=model, num_searches=5)
        examples = parallel_self_play(game, mcts, n_games=4, max_moves=None, n_workers=2)
        assert isinstance(examples, list)
        assert len(examples) >= 4

    def test_multi_worker_outcomes_valid(self, game, model):
        mcts = MCTS(game, model=model, num_searches=5)
        examples = parallel_self_play(game, mcts, n_games=4, max_moves=None, n_workers=2)
        for _, _, outcome in examples:
            assert outcome in {-1.0, 0.0, 1.0}

    def test_requires_model_in_mcts(self, game):
        """parallel_self_play must raise if MCTS has no model."""
        mcts = MCTS(game, model=None, num_searches=5)
        with pytest.raises(ValueError, match="model"):
            parallel_self_play(game, mcts, n_games=1, n_workers=1)


def _make_batch(game, model, batch_size):
    """Create a random batch of training examples."""
    encoded_states = np.random.rand(batch_size, 3, game.row_count, game.column_count).astype(np.float32)
    policies = np.random.dirichlet(np.ones(game.action_size), size=batch_size).astype(np.float32)
    outcomes = np.random.choice([-1.0, 0.0, 1.0], size=batch_size).astype(np.float32)
    return encoded_states, policies, outcomes
