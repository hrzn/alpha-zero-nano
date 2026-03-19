"""Tests for the standalone chess training script (run_training.py)."""

import os

import pytest
import torch
import torch.optim as optim

from chess_game.chess_game import ChessGame
from model.model import ResNet
from train.run_training import load_latest_checkpoint, save_checkpoint


@pytest.fixture
def game():
    return ChessGame()


@pytest.fixture
def small_args():
    return {
        "num_res_blocks": 1,
        "num_hidden": 32,
        "num_searches": 5,
        "num_self_play_games": 1,
        "max_moves": 5,
        "num_epochs": 1,
        "batch_size": 8,
        "lr": 1e-3,
        "num_iterations": 2,
        "checkpoint_interval": 1,
    }


@pytest.fixture
def small_model(game):
    return ResNet(game, num_res_blocks=1, num_hidden=32)


@pytest.fixture
def optimizer(small_model):
    return optim.Adam(small_model.parameters(), lr=1e-3)


class TestCheckpointSave:
    def test_produces_file_at_given_path(self, small_model, optimizer, small_args, tmp_path):
        path = str(tmp_path / "test.pt")
        save_checkpoint(path, iteration=3, model=small_model, optimizer=optimizer,
                        loss_history=[0.9, 0.7], args=small_args)
        assert os.path.exists(path)

    def test_contains_expected_keys(self, small_model, optimizer, small_args, tmp_path):
        path = str(tmp_path / "test.pt")
        save_checkpoint(path, iteration=5, model=small_model, optimizer=optimizer,
                        loss_history=[0.5, 0.4], args=small_args)

        checkpoint = torch.load(path, weights_only=False)
        for key in ("iteration", "model_state_dict", "optimizer_state_dict",
                    "loss_history", "args"):
            assert key in checkpoint

    def test_iteration_stored_correctly(self, small_model, optimizer, small_args, tmp_path):
        path = str(tmp_path / "test.pt")
        save_checkpoint(path, iteration=7, model=small_model, optimizer=optimizer,
                        loss_history=[], args=small_args)
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["iteration"] == 7

    def test_loss_history_stored_correctly(self, small_model, optimizer, small_args, tmp_path):
        path = str(tmp_path / "test.pt")
        losses = [1.0, 0.8, 0.6]
        save_checkpoint(path, iteration=3, model=small_model, optimizer=optimizer,
                        loss_history=losses, args=small_args)
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["loss_history"] == losses


class TestCheckpointLoad:
    def test_no_checkpoint_returns_none(self, game, small_args, tmp_path):
        model, optimizer, iteration, loss_history = load_latest_checkpoint(
            str(tmp_path), game, small_args
        )
        assert model is None
        assert optimizer is None
        assert iteration is None
        assert loss_history == []

    def test_resume_restores_iteration(self, game, small_model, optimizer, small_args, tmp_path):
        path = str(tmp_path / "chess_iter_0005.pt")
        save_checkpoint(path, iteration=5, model=small_model, optimizer=optimizer,
                        loss_history=[0.9], args=small_args)

        _, _, loaded_iter, _ = load_latest_checkpoint(str(tmp_path), game, small_args)
        assert loaded_iter == 5

    def test_resume_restores_loss_history(self, game, small_model, optimizer, small_args, tmp_path):
        losses = [1.0, 0.8, 0.6]
        path = str(tmp_path / "chess_iter_0003.pt")
        save_checkpoint(path, iteration=3, model=small_model, optimizer=optimizer,
                        loss_history=losses, args=small_args)

        _, _, _, loaded_losses = load_latest_checkpoint(str(tmp_path), game, small_args)
        assert loaded_losses == losses

    def test_resume_restores_model_weights(self, game, small_model, optimizer, small_args, tmp_path):
        path = str(tmp_path / "chess_iter_0001.pt")
        save_checkpoint(path, iteration=1, model=small_model, optimizer=optimizer,
                        loss_history=[], args=small_args)

        loaded_model, _, _, _ = load_latest_checkpoint(str(tmp_path), game, small_args)
        for p1, p2 in zip(small_model.parameters(), loaded_model.parameters()):
            assert torch.equal(p1, p2)

    def test_loads_highest_numbered_checkpoint(self, game, small_args, tmp_path):
        """When multiple checkpoints exist, load the one with the highest iteration."""
        for n in [1, 5, 3]:
            m = ResNet(game, num_res_blocks=1, num_hidden=32)
            opt = optim.Adam(m.parameters(), lr=1e-3)
            path = str(tmp_path / f"chess_iter_{n:04d}.pt")
            save_checkpoint(path, iteration=n, model=m, optimizer=opt,
                            loss_history=[float(n)], args=small_args)

        _, _, loaded_iter, _ = load_latest_checkpoint(str(tmp_path), game, small_args)
        assert loaded_iter == 5

    def test_training_continues_from_resumed_iteration(self, game, small_model, optimizer,
                                                        small_args, tmp_path):
        """After loading a checkpoint at iteration N, training should start at N, not 0."""
        path = str(tmp_path / "chess_iter_0003.pt")
        save_checkpoint(path, iteration=3, model=small_model, optimizer=optimizer,
                        loss_history=[1.0, 0.9, 0.8], args=small_args)

        _, _, loaded_iter, loaded_losses = load_latest_checkpoint(
            str(tmp_path), game, small_args
        )
        # Simulate starting from loaded_iter
        start = loaded_iter
        assert start == 3
        assert len(loaded_losses) == 3
