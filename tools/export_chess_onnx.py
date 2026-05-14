"""Export a chess ResNet to ONNX, with a dynamic batch axis.

Distinct from ``tools/export_for_web.py`` (which emits a custom
``weights.bin``+``weights.json`` pair for the hand-rolled TS forward pass at
``web/src/model.ts``). The Rust port will load this ONNX file via an ONNX
runtime (``ort``/``tch``) in a later phase; Phase 0 only verifies the
artifact is well-formed and numerically faithful via the companion pytest
``tools/test_export_chess_onnx.py``.

Usage:
    uv run python tools/export_chess_onnx.py                          # fresh-init model, S-shape
    uv run python tools/export_chess_onnx.py --ckpt path/to/chess.pt  # real checkpoint
"""

import argparse
import os

import torch

from chess_game.chess_game import ChessGame
from model.model import ResNet

# S-preset shape from train/run_training.py; the canonical "chess training is
# feasible" model size on M1. XS and M can re-export with --num-res-blocks /
# --num-hidden if needed.
DEFAULT_NUM_RES_BLOCKS = 5
DEFAULT_NUM_HIDDEN = 128
DEFAULT_OPSET = 17
DEFAULT_OUT = "checkpoints/chess/model.onnx"


class ChessForward(torch.nn.Module):
    """Wraps ResNet so the ONNX output value tensor has shape (B, 1) instead
    of (B,). Keeps downstream Rust code working with a fixed-rank output."""

    def __init__(self, inner: ResNet):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        policy_logits, value = self.inner(x)
        return policy_logits, value.unsqueeze(-1)


def build_model(num_res_blocks: int, num_hidden: int, ckpt_path: str | None) -> torch.nn.Module:
    game = ChessGame()
    if ckpt_path is None:
        torch.manual_seed(0)
    model = ResNet(game, num_res_blocks=num_res_blocks, num_hidden=num_hidden)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd)
    wrapper = ChessForward(model)
    wrapper.eval()  # propagates to children — BN uses running stats, dropout off
    return wrapper


def export(model: torch.nn.Module, out_path: str, opset: int = DEFAULT_OPSET) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dummy = torch.zeros(1, 17, 8, 8, dtype=torch.float32)
    # Use the legacy TorchScript-based exporter (dynamo=False) so we don't
    # need onnxscript. The output is standard ONNX consumable by onnxruntime,
    # ort (Rust), and tch.
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["state"],
        output_names=["policy_logits", "value"],
        opset_version=opset,
        dynamic_axes={
            "state": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", default=None, help="Optional .pt checkpoint to load")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--num-res-blocks", type=int, default=DEFAULT_NUM_RES_BLOCKS)
    p.add_argument("--num-hidden", type=int, default=DEFAULT_NUM_HIDDEN)
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    args = p.parse_args()

    model = build_model(args.num_res_blocks, args.num_hidden, args.ckpt)
    export(model, args.out, opset=args.opset)
    size = os.path.getsize(args.out)
    src = args.ckpt or "fresh-init (seed=0)"
    print(
        f"Wrote {args.out}  ({size:,} bytes; opset={args.opset}; "
        f"{args.num_res_blocks}×{args.num_hidden}; source={src})"
    )


if __name__ == "__main__":
    main()
