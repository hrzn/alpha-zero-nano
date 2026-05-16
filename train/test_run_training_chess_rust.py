"""Integration test for the Rust-backed chess training driver.

Runs a 2-iteration tiny config end-to-end:
  - builds the Rust selfplay binary if needed
  - exports ONNX, generates shards, trains, saves checkpoint
  - resumes from the checkpoint on a second invocation

Skipped automatically when:
  - cargo is not on PATH
  - the venv's libonnxruntime is missing
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _have_cargo() -> bool:
    return shutil.which("cargo") is not None


def _have_libonnxruntime() -> bool:
    candidate = (
        REPO_ROOT
        / ".venv"
        / "lib"
        / "python3.12"
        / "site-packages"
        / "onnxruntime"
        / "capi"
    )
    if not candidate.exists():
        return False
    return any(candidate.glob("libonnxruntime*"))


pytestmark = [
    pytest.mark.skipif(not _have_cargo(), reason="cargo not on PATH"),
    pytest.mark.skipif(
        not _have_libonnxruntime(),
        reason="libonnxruntime not installed — run `uv sync --group dev`",
    ),
]


@pytest.fixture
def tiny_preset():
    """A 2-iteration, 2-game, very-small preset used only by this test —
    keeps end-to-end runtime under ~30s on a laptop."""
    return {
        "_description": "test-only preset, do not use for real training",
        "game": "chess",
        "num_res_blocks": 2,
        "num_hidden": 32,
        "num_searches": 8,
        "selfplay_batch_size": 4,
        "c_puct": 1.0,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "num_self_play_games": 2,
        "num_workers": 2,
        "max_moves": 20,
        "temp_threshold": 4,
        "num_epochs": 1,
        "train_batch_size": 16,
        "lr": 1e-3,
        "lr_milestones": [],
        "replay_buffer_size": 500,
        "num_iterations": 2,
        "checkpoint_interval": 1,
        "eval_interval": 99,  # disable arena for the test
        "eval_games": 2,
        "eval_searches": 8,
    }


def test_end_to_end_two_iterations(tiny_preset, monkeypatch):
    """Run the driver for two iterations with a tiny preset. Asserts:
      - 1+ checkpoints written
      - loss_history.json present and finite
      - resume from the saved checkpoint picks up at the right iter
    """
    from train import run_training_chess_rust as driver

    monkeypatch.setitem(driver.PRESETS, "TEST", tiny_preset)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        driver.run_training(
            preset_name="TEST", device="cpu", checkpoint_dir=str(run_dir),
        )

        # Checkpoints written.
        ckpts = sorted(run_dir.glob("chess_iter_*.pt"))
        assert len(ckpts) >= 2, f"expected >= 2 checkpoints, got {len(ckpts)}"
        assert (run_dir / "chess_best.pt").exists()

        # Loss history is finite.
        with open(run_dir / "loss_history.json") as f:
            losses = json.load(f)
        assert len(losses) == 2
        for v in losses:
            assert v == v and v != float("inf"), f"non-finite loss {v}"

        # Shards landed where we expect.
        shard_dirs = sorted((run_dir / "shards").iterdir())
        assert len(shard_dirs) == 2
        for d in shard_dirs:
            for name in ("states.npy", "policies.npy", "values.npy"):
                assert (d / name).exists(), f"missing {name} in {d}"

        # ONNX exports too.
        onnx_files = sorted((run_dir / "onnx").glob("iter_*.onnx"))
        assert len(onnx_files) == 2

        # Resume: bump iteration count, rerun — should pick up at iter 2 and
        # produce iter 3.
        tiny_preset["num_iterations"] = 3
        driver.run_training(
            preset_name="TEST", device="cpu", checkpoint_dir=str(run_dir),
        )
        ckpts = sorted(run_dir.glob("chess_iter_*.pt"))
        assert any("0003" in p.name for p in ckpts), \
            f"resume did not produce iter-3 checkpoint, got {[p.name for p in ckpts]}"
