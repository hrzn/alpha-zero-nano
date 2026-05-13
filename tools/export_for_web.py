"""Export a trained checkpoint to a browser-loadable format.

Produces two files in --out-dir:
  weights.bin   — all tensors from the model state_dict, concatenated as
                  little-endian float32. No padding between tensors.
  weights.json  — manifest: {tensors: [{name, shape, offset, size}, ...],
                  model_args: {...}, game: {...}, bn_eps: float}

The browser-side TS port reads the manifest, fetches the blob, and slices
it into Float32Arrays at the offsets given. Floats are float32 (4 bytes
each); offsets and sizes are byte-counted.

Usage:
    uv run python -m tools.export_for_web                    # latest C4 ckpt
    uv run python -m tools.export_for_web --ckpt path.pt
    uv run python -m tools.export_for_web --out-dir web/public
"""

import argparse
import glob
import json
import os
import struct

import torch

DEFAULT_CHAMPION = "checkpoints/c4/connect4_best.pt"
DEFAULT_CKPT_GLOB = "checkpoints/c4/connect4_iter_*.pt"
DEFAULT_OUT_DIR = "web/public"
BN_EPS = 1e-5  # PyTorch BatchNorm2d default


def find_default_checkpoint():
    """Prefer the arena-gated champion if present, else fall back to the
    most recent iter checkpoint. Letting the web demo always ship the
    best-validated model is the whole point of arena gating."""
    if os.path.exists(DEFAULT_CHAMPION):
        return DEFAULT_CHAMPION
    files = sorted(glob.glob(DEFAULT_CKPT_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No champion at {DEFAULT_CHAMPION} and no iter checkpoints matching {DEFAULT_CKPT_GLOB}"
        )
    return files[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt",
        default=None,
        help=f"Checkpoint .pt (default: {DEFAULT_CHAMPION} if present, else latest {DEFAULT_CKPT_GLOB})",
    )
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    ckpt_path = args.ckpt or find_default_checkpoint()
    print(f"Loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    os.makedirs(args.out_dir, exist_ok=True)
    bin_path = os.path.join(args.out_dir, "weights.bin")
    manifest_path = os.path.join(args.out_dir, "weights.json")

    tensors_meta = []
    offset = 0
    with open(bin_path, "wb") as f:
        for name, t in state_dict.items():
            # Skip BN's num_batches_tracked — irrelevant for inference.
            if name.endswith("num_batches_tracked"):
                continue
            arr = t.detach().to(torch.float32).contiguous().cpu().numpy()
            data = arr.tobytes()
            f.write(data)
            tensors_meta.append(
                {
                    "name": name,
                    "shape": list(arr.shape),
                    "offset": offset,
                    "size": len(data),
                }
            )
            offset += len(data)

    manifest = {
        "tensors": tensors_meta,
        "model_args": {
            "num_res_blocks": model_args.get("num_res_blocks", 3),
            "num_hidden": model_args.get("num_hidden", 64),
        },
        "game": {
            "row_count": 6,
            "column_count": 7,
            "action_size": 7,
            "num_channels": 3,
        },
        "bn_eps": BN_EPS,
        "source_checkpoint": os.path.basename(ckpt_path),
        "iteration": ckpt.get("iteration", None) if isinstance(ckpt, dict) else None,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_floats = offset // 4
    print(f"Wrote {bin_path}  ({offset:,} bytes, {total_floats:,} float32)")
    print(f"Wrote {manifest_path}  ({len(tensors_meta)} tensors)")
    print()
    print("Tensor summary:")
    for meta in tensors_meta:
        shape_str = "×".join(str(d) for d in meta["shape"])
        print(f"  {meta['name']:50s}  {shape_str:20s}  {meta['size']:>10,} bytes")


if __name__ == "__main__":
    main()
