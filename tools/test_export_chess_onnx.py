"""Phase 0 sanity test: export a chess ResNet to ONNX and verify that
onnxruntime's outputs agree with PyTorch's eager forward to within 1e-4 on
random (17,8,8) inputs.

Tolerance mirrors the existing web agreement test at
``web/tools/agreement_test.ts`` (1e-4) — float32 round-trip plus the cumulative
op count of a small ResNet typically drifts in the 1e-6..1e-5 range, so 1e-4
leaves headroom.

This test does NOT depend on the Rust crate. It only proves the ONNX artifact
is well-formed. Phase 3 (separately planned) will exercise the ONNX file from
Rust via `ort` or `tch`.
"""

import os
import sys
import tempfile

import numpy as np
import onnxruntime as ort
import pytest
import torch

# `tools/` is not a Python package; add it to sys.path so we can import the
# sibling export module without needing an __init__.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_chess_onnx import build_model, export  # noqa: E402

# Smaller shape than the S-preset to keep the test fast — the export pipeline
# is shape-agnostic, so a 3-block / 64-hidden model exercises the same code
# path as 5/128 in a fraction of the time.
TEST_NUM_RES_BLOCKS = 3
TEST_NUM_HIDDEN = 64
BATCH = 16
TOL = 1e-4


@pytest.fixture(scope="module")
def exported_model():
    """Export once, reuse across asserts."""
    model = build_model(TEST_NUM_RES_BLOCKS, TEST_NUM_HIDDEN, ckpt_path=None)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "chess.onnx")
        export(model, path)
        yield model, path


def test_exported_file_is_nonempty(exported_model):
    _, path = exported_model
    assert os.path.getsize(path) > 0


def test_pytorch_vs_ort_agreement(exported_model):
    model, path = exported_model
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    x_np = rng.standard_normal((BATCH, 17, 8, 8), dtype=np.float32)

    # PyTorch forward
    with torch.no_grad():
        torch_logits, torch_value = model(torch.from_numpy(x_np))
    torch_logits_np = torch_logits.cpu().numpy()
    torch_value_np = torch_value.cpu().numpy()

    # ORT forward
    ort_outputs = sess.run(["policy_logits", "value"], {"state": x_np})
    ort_logits, ort_value = ort_outputs

    assert torch_logits_np.shape == ort_logits.shape == (BATCH, 4096), (
        f"policy_logits shape {torch_logits_np.shape}/{ort_logits.shape}"
    )
    assert torch_value_np.shape == ort_value.shape == (BATCH, 1), (
        f"value shape {torch_value_np.shape}/{ort_value.shape}"
    )

    logits_diff = float(np.max(np.abs(torch_logits_np - ort_logits)))
    value_diff = float(np.max(np.abs(torch_value_np - ort_value)))
    assert logits_diff < TOL, f"policy_logits max abs diff {logits_diff} >= {TOL}"
    assert value_diff < TOL, f"value max abs diff {value_diff} >= {TOL}"


def test_dynamic_batch_axis(exported_model):
    """The export sets a dynamic batch axis — verify ORT accepts different
    batch sizes (1 and 7), not just the one used at export time."""
    _, path = exported_model
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch in (1, 7):
        x = np.zeros((batch, 17, 8, 8), dtype=np.float32)
        logits, value = sess.run(["policy_logits", "value"], {"state": x})
        assert logits.shape == (batch, 4096)
        assert value.shape == (batch, 1)
