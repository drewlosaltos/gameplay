#!/usr/bin/env python3
"""Export a four-frame SkyBall gameplay classifier to portable ONNX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx
import torch

REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyball_gameplay.modeling import load_checkpoint_model as load_model
from skyball_gameplay.modeling import sha256_file


DEFAULT_CHECKPOINT = Path(
    "/mnt/t/output/trn/skyball_gameplay/"
    "v5_siglip_base_f4_avg_h512w512_step4s2_lr1e4_blr5e6_20260719/"
    "best_siglip_base_f4_avg_binary_h512w512_step4s2_pre_drop010.pth.tar"
)


class GameplayExportWrapper(torch.nn.Module):
    """Expose classifier logits and gameplay probability for runtime use."""

    def __init__(
        self,
        model: torch.nn.Module,
        gameplay_class_index: int,
    ) -> None:
        """Wrap a temporal classifier with its gameplay score semantics.

        Args:
            model: Loaded temporal gameplay classifier.
            gameplay_class_index: Output class index representing gameplay.
        """
        super().__init__()
        self.model = model
        self.gameplay_class_index = gameplay_class_index

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two-class logits and the gameplay softmax probability.

        Args:
            frames: Normalized RGB tensor shaped ``[B, 4, 3, H, W]``.

        Returns:
            Raw class logits and one gameplay probability per batch item.
        """
        logits = self.model(frames)
        gameplay_score = torch.softmax(logits, dim=1)[:, self.gameplay_class_index]
        return logits, gameplay_score


def runtime_contract(
    model_metadata: dict[str, Any],
    batch_size: int,
    dynamic_batch: bool,
    opset: int,
) -> dict[str, Any]:
    """Build the machine-readable ONNX input and output contract.

    Args:
        model_metadata: Metadata returned by the checkpoint loader.
        batch_size: Batch size used for tracing and fixed-shape exports.
        dynamic_batch: Whether the ONNX batch dimension is symbolic.
        opset: ONNX opset used for export.

    Returns:
        Portable runtime contract suitable for the publication sidecar.
    """
    image_height, image_width = [int(value) for value in model_metadata["image_size"]]
    input_frames = int(model_metadata["input_frames"])
    leading_dimension: int | str = "batch" if dynamic_batch else batch_size
    classes = list(model_metadata["classes"])
    gameplay_class_index = int(model_metadata["gameplay_class_index"])
    return {
        "opset": opset,
        "input": {
            "name": "frames",
            "shape": [
                leading_dimension,
                input_frames,
                3,
                image_height,
                image_width,
            ],
            "dtype": "float32",
            "color_mode": "RGB",
            "tensor_scale": "float_[0,1]_then_normalize",
            "normalization_mean": list(model_metadata["normalization_mean"]),
            "normalization_std": list(model_metadata["normalization_std"]),
            "resize": {
                "height": image_height,
                "width": image_width,
                "mode": "stretch_bilinear_antialias",
            },
        },
        "outputs": {
            "logits": {
                "shape": [leading_dimension, len(classes)],
                "classes": classes,
            },
            "gameplay_score": {
                "shape": [leading_dimension],
                "definition": (
                    f"softmax(logits) class index {gameplay_class_index} "
                    f"({classes[gameplay_class_index]})"
                ),
                "decision": "gameplay when score >= gate threshold",
            },
        },
        "temporal_sampling": {
            "frames": input_frames,
            "frame_spacing_seconds": 1.0,
            "window_duration_seconds": float(input_frames),
        },
        "dynamic_batch": dynamic_batch,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a deterministic JSON artifact.

    Args:
        path: Destination JSON path.
        payload: Serializable object to write.
    """
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ONNX export.

    Returns:
        Parsed export arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--fixed-batch",
        action="store_true",
        help="Export the traced batch size instead of a symbolic batch dimension.",
    )
    return parser.parse_args()


def main() -> None:
    """Export the registered gameplay model and write its runtime sidecar."""
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists():
        raise FileExistsError(f"Refusing existing ONNX output: {args.output}")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    if sidecar.exists():
        raise FileExistsError(f"Refusing existing ONNX sidecar: {sidecar}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    wrapper = GameplayExportWrapper(
        model=model,
        gameplay_class_index=int(metadata["gameplay_class_index"]),
    ).to(device)
    wrapper.eval()
    image_height, image_width = [int(value) for value in metadata["image_size"]]
    torch.manual_seed(20260719)
    example = torch.randn(
        args.batch_size,
        int(metadata["input_frames"]),
        3,
        image_height,
        image_width,
        dtype=torch.float32,
        device=device,
    )
    dynamic_batch = not args.fixed_batch
    dynamic_axes = (
        {
            "frames": {0: "batch"},
            "logits": {0: "batch"},
            "gameplay_score": {0: "batch"},
        }
        if dynamic_batch
        else None
    )
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            example,
            str(args.output),
            input_names=["frames"],
            output_names=["logits", "gameplay_score"],
            opset_version=args.opset,
            export_params=True,
            keep_initializers_as_inputs=False,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
            external_data=False,
        )
    onnx_model = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(onnx_model)
    contract = runtime_contract(
        model_metadata=metadata,
        batch_size=args.batch_size,
        dynamic_batch=dynamic_batch,
        opset=args.opset,
    )
    write_json(
        sidecar,
        {
            "schema_version": 1,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "onnx": str(args.output),
            "onnx_sha256": sha256_file(args.output),
            "onnx_size_bytes": args.output.stat().st_size,
            "torch_version": torch.__version__,
            "onnx_version": onnx.__version__,
            "model": metadata,
            "runtime_contract": contract,
        },
    )
    print(f"wrote {args.output}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
