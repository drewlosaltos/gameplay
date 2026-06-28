#!/usr/bin/env python3
"""Export the SkyBall large addcats detector to ONNX."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F

from skyball_trt_common import (
    DEFAULT_CHECKPOINT,
    FULL_OUTPUT_NAMES,
    default_artifact_dir,
    load_export_model,
    make_input,
    write_json,
)
from rfdetr.models.backbone.projector import LayerNorm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-state", default="model")
    parser.add_argument(
        "--output-mode",
        choices=("center", "all-frames"),
        default="center",
        help="Export center-frame boxes/logits or full tracking outputs.",
    )
    parser.add_argument(
        "--keep-interpolate-antialias",
        action="store_true",
        help=(
            "Keep antialias=True during ONNX export. TensorRT/PyTorch ONNX "
            "currently cannot export the antialiased bicubic op used by DINOv2."
        ),
    )
    return parser.parse_args()


@contextmanager
def export_compatibility_patches() -> Iterator[None]:
    """Patch narrow export-only cases that the ONNX exporter cannot lower."""
    original_interpolate = F.interpolate
    original_layer_norm_forward = LayerNorm.forward

    def patched_interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None, **kwargs):
        if kwargs.get("antialias") and mode in {"bilinear", "bicubic"}:
            kwargs = dict(kwargs)
            kwargs["antialias"] = False
        return original_interpolate(input, size, scale_factor, mode, align_corners, **kwargs)

    def patched_layer_norm_forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)

    F.interpolate = patched_interpolate
    LayerNorm.forward = patched_layer_norm_forward
    try:
        yield
    finally:
        F.interpolate = original_interpolate
        LayerNorm.forward = original_layer_norm_forward


def main() -> None:
    args = parse_args()
    output_suffix = "" if args.output_mode == "center" else "_allframes"
    output = args.output or (
        default_artifact_dir()
        / (
            f"large_addcats_b{args.batch_size}_t{args.num_frames}_"
            f"{args.resolution}{output_suffix}.onnx"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    model, saved_args = load_export_model(
        args.checkpoint,
        device=args.device,
        checkpoint_state=args.checkpoint_state,
        output_mode=args.output_mode,
    )
    dummy = make_input(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        resolution=args.resolution,
        device=args.device,
        dtype=torch.float32,
    )

    with torch.inference_mode():
        outputs = model(dummy)
    output_names = (
        ["boxes", "logits"]
        if args.output_mode == "center"
        else list(FULL_OUTPUT_NAMES)
    )
    output_shapes = {
        name: list(tensor.shape)
        for name, tensor in zip(output_names, outputs)
    }
    print("pytorch outputs:")
    for name, tensor in zip(output_names, outputs):
        print(f"  {name}={tuple(tensor.shape)} {tensor.dtype}")

    export_context = nullcontext() if args.keep_interpolate_antialias else export_compatibility_patches()
    with export_context:
        # Use the legacy exporter because this repository already carries symbolic
        # support for RF-DETR's custom/deformable ops on that path.
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=["frames"],
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            opset_version=args.opset,
            dynamic_axes=None,
            verbose=False,
            dynamo=False,
        )
    sidecar = output.with_suffix(output.suffix + ".json")
    write_json(
        sidecar,
        {
            "checkpoint": str(args.checkpoint),
            "onnx": str(output),
            "batch_size": args.batch_size,
            "num_frames": args.num_frames,
            "resolution": args.resolution,
            "output_mode": args.output_mode,
            "opset": args.opset,
            "interpolate_antialias": bool(args.keep_interpolate_antialias),
            "export_compatibility_patches": not args.keep_interpolate_antialias,
            "class_names": list(getattr(saved_args, "class_names", []) or []),
            "output_names": output_names,
            "output_shapes": output_shapes,
        },
    )
    print(f"wrote {output}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
