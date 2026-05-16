"""Emit ONNX model + Python reference data for Rust-side ORT parity test.

For Phase 3 we want to know that calling the exported ONNX through ort (Rust)
matches PyTorch's eager forward on the same inputs, just as
``tools/test_export_chess_onnx.py`` proved for onnxruntime-python in Phase 0.

This script:
  1. Builds a fresh-init chess ResNet (seed=0, small shape).
  2. Exports it to ``rust/tests/fixtures/chess_inference.onnx`` via the same
     code path as ``tools/export_chess_onnx.py``.
  3. Generates N random ``(17, 8, 8)`` inputs, runs PyTorch eager forward in
     eval mode, and writes the resulting (policy_logits, value) tuples next
     to the ONNX file as JSON. Logits are stored, **not** softmax probs —
     Rust will softmax them too, so we compare the raw model output.

Usage:
    uv run python tools/gen_inference_parity_fixtures.py
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_chess_onnx import build_model, export  # noqa: E402

SCHEMA_VERSION = 1
DEFAULT_ONNX_OUT = "rust/tests/fixtures/chess_inference.onnx"
DEFAULT_JSON_OUT = "rust/tests/fixtures/chess_inference.json"

NUM_RES_BLOCKS = 3
NUM_HIDDEN = 64
MODEL_SEED = 0
NUM_SAMPLES = 16
RNG_SEED = 42


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--onnx-out", default=DEFAULT_ONNX_OUT)
    p.add_argument("--json-out", default=DEFAULT_JSON_OUT)
    p.add_argument("--n", type=int, default=NUM_SAMPLES)
    args = p.parse_args()

    # Export the ONNX. `build_model` already sets eval mode + seeds torch
    # when ckpt_path is None.
    model = build_model(NUM_RES_BLOCKS, NUM_HIDDEN, ckpt_path=None)
    os.makedirs(os.path.dirname(args.onnx_out) or ".", exist_ok=True)
    export(model, args.onnx_out)

    # Generate reference inputs and PyTorch eager outputs. Same RNG seed as
    # the Phase 0 onnxruntime-vs-PyTorch test for repeatability.
    rng = np.random.default_rng(RNG_SEED)
    inputs = rng.standard_normal((args.n, 17, 8, 8), dtype=np.float32)

    with torch.no_grad():
        logits, value = model(torch.from_numpy(inputs))
    logits_np = logits.cpu().numpy()  # (n, 4096)
    value_np = value.cpu().numpy()    # (n, 1)

    samples = []
    for i in range(args.n):
        samples.append(
            {
                "input": inputs[i].flatten().tolist(),       # 1088 floats
                "expected_policy_logits": logits_np[i].tolist(),  # 4096 floats
                "expected_value": float(value_np[i, 0]),
            }
        )

    out = {
        "schema_version": SCHEMA_VERSION,
        "onnx_path": os.path.relpath(args.onnx_out, os.path.dirname(args.json_out)),
        "model": {
            "num_res_blocks": NUM_RES_BLOCKS,
            "num_hidden": NUM_HIDDEN,
            "seed": MODEL_SEED,
        },
        "input_shape": [17, 8, 8],
        "samples": samples,
    }

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(out, f)

    onnx_kb = os.path.getsize(args.onnx_out) / 1024
    json_mb = os.path.getsize(args.json_out) / 1e6
    print(f"Wrote {args.onnx_out}  ({onnx_kb:.0f} KB)")
    print(f"Wrote {args.json_out}  ({len(samples)} samples, {json_mb:.1f} MB)")


if __name__ == "__main__":
    main()
