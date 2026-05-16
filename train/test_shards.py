"""Integration: invoke the Rust ``selfplay`` binary and verify Python loads
the resulting shards into well-shaped numpy arrays.

Skipped automatically when:
  - The Rust toolchain isn't on PATH.
  - The Phase 3 chess_inference.onnx fixture is missing (regenerate with
    ``uv run python tools/gen_inference_parity_fixtures.py``).

The point of this test isn't algorithmic correctness — that's covered by the
Rust-side selfplay smoke. The point is the cross-language boundary: the
exact byte format Rust writes and Python reads.
"""

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX_FIXTURE = os.path.join(
    REPO_ROOT, "rust", "tests", "fixtures", "chess_inference.onnx"
)


def _have_cargo() -> bool:
    return shutil.which("cargo") is not None


@pytest.mark.skipif(not _have_cargo(), reason="cargo not on PATH")
@pytest.mark.skipif(
    not os.path.exists(ONNX_FIXTURE),
    reason="chess_inference.onnx missing — regenerate via "
    "tools/gen_inference_parity_fixtures.py",
)
def test_selfplay_binary_roundtrip():
    """Run the Rust selfplay binary for 2 short games, then load the
    shards in Python and check shape, dtype, and value bounds."""
    from train.shards import load_shard  # local import so the file is also valid without numpy

    rust_dir = os.path.join(REPO_ROOT, "rust")
    with tempfile.TemporaryDirectory() as tmp:
        # Build + run in release for speed; the binary is already cached
        # after first compile.
        out_dir = os.path.join(tmp, "shards")
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            "cargo", "run", "--release", "--quiet", "--bin", "selfplay", "--",
            "--model", ONNX_FIXTURE,
            "--out", out_dir,
            "--num-games", "2",
            "--num-workers", "2",
            "--num-searches", "16",
            "--batch-size", "4",
            "--max-moves", "20",
            "--temp-threshold", "4",
            "--dirichlet-alpha", "0.3",
            "--dirichlet-epsilon", "0.25",
            "--seed", "1",
        ]
        result = subprocess.run(
            cmd, cwd=rust_dir, capture_output=True, text=True, timeout=180,
        )
        if result.returncode != 0:
            print("STDOUT:", result.stdout, file=sys.stderr)
            print("STDERR:", result.stderr, file=sys.stderr)
            pytest.fail(f"selfplay binary failed: rc={result.returncode}")

        shard = load_shard(out_dir, mmap=False)
        assert len(shard) > 0, "binary wrote zero examples"
        # max_moves=20, 2 games → at most 40 examples.
        assert len(shard) <= 40, f"got {len(shard)} examples for 2x20-move games"

        # Per-example sanity.
        assert shard.states.shape[1:] == (17, 8, 8)
        assert shard.policies.shape[1] == 4096
        assert np.all(np.isfinite(shard.states))
        assert np.all(np.isfinite(shard.policies))
        assert np.all(np.isfinite(shard.values))

        # Policies should be probability distributions.
        sums = shard.policies.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)
        assert (shard.policies >= 0.0).all()

        # Values within [-1, 1].
        assert (np.abs(shard.values) <= 1.0 + 1e-5).all()
