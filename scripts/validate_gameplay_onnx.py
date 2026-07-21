#!/usr/bin/env python3
"""Validate PyTorch/ONNX gameplay parity on a real validation window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy
import onnxruntime
import torch
from PIL import Image

REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyball_gameplay.modeling import load_checkpoint_model as load_model
from skyball_gameplay.modeling import sha256_file
from export_gameplay_onnx import DEFAULT_CHECKPOINT
from export_gameplay_onnx import GameplayExportWrapper


DEFAULT_DATASET_ROOT = Path("/mnt/t/data/vball/skyball/gameplay/v5")


def load_validation_window(
    dataset_root: Path,
    metadata: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and preprocess the first deterministic validation window.

    Args:
        dataset_root: Root containing images and ``metadata/windows.jsonl``.
        metadata: Checkpoint preprocessing metadata.

    Returns:
        A batch-one input tensor and its source manifest record.
    """
    manifest = dataset_root / "metadata" / "windows.jsonl"
    selected: dict[str, Any] | None = None
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") == "val" and len(row.get("frame_paths", [])) == 4:
                selected = row
                break
    if selected is None:
        raise ValueError(f"No four-frame validation window in {manifest}")
    arrays = []
    for relative_path in selected["frame_paths"]:
        with Image.open(dataset_root / relative_path) as image:
            arrays.append(numpy.asarray(image.convert("RGB"), dtype=numpy.uint8).copy())
    frames = torch.stack(
        [torch.from_numpy(array).permute(2, 0, 1) for array in arrays]
    ).float()
    frames = frames.div(255.0)
    frames = torch.nn.functional.interpolate(
        frames,
        size=tuple(int(value) for value in metadata["image_size"]),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    mean = torch.tensor(metadata["normalization_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(metadata["normalization_std"]).view(1, 3, 1, 1)
    return ((frames - mean) / std).unsqueeze(0), selected


def difference_metrics(reference: numpy.ndarray, candidate: numpy.ndarray) -> dict[str, float]:
    """Return absolute numerical differences between two output arrays.

    Args:
        reference: Reference PyTorch output.
        candidate: Candidate ONNX output.

    Returns:
        Maximum and mean absolute error.
    """
    difference = numpy.abs(reference.astype(numpy.float64) - candidate.astype(numpy.float64))
    return {
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()) if difference.size else 0.0,
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed parity-validation arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--provider",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument("--gate-threshold", type=float, default=0.2)
    parser.add_argument("--max-logit-error", type=float, default=1e-3)
    parser.add_argument("--max-score-error", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    """Compare the exported graph to its source checkpoint and write evidence."""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing parity output: {args.output}")
    if not args.checkpoint.is_file() or not args.onnx.is_file():
        raise FileNotFoundError("checkpoint and ONNX paths must exist")
    device = torch.device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    wrapper = GameplayExportWrapper(
        model=model,
        gameplay_class_index=int(metadata["gameplay_class_index"]),
    ).to(device)
    wrapper.eval()
    inputs, source = load_validation_window(args.dataset_root, metadata)
    with torch.inference_mode():
        torch_logits, torch_score = wrapper(inputs.to(device))
    torch_logits_array = torch_logits.cpu().numpy()
    torch_score_array = torch_score.cpu().numpy()

    available = onnxruntime.get_available_providers()
    if args.provider == "cuda" or (
        args.provider == "auto" and "CUDAExecutionProvider" in available
    ):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(args.onnx), providers=providers)
    onnx_logits, onnx_score = session.run(
        ["logits", "gameplay_score"],
        {"frames": inputs.numpy()},
    )
    logit_difference = difference_metrics(torch_logits_array, onnx_logits)
    score_difference = difference_metrics(torch_score_array, onnx_score)
    class_agreement = bool(
        numpy.array_equal(torch_logits_array.argmax(axis=1), onnx_logits.argmax(axis=1))
    )
    gate_agreement = bool(
        numpy.array_equal(
            torch_score_array >= args.gate_threshold,
            onnx_score >= args.gate_threshold,
        )
    )
    passed = (
        logit_difference["max_abs"] <= args.max_logit_error
        and score_difference["max_abs"] <= args.max_score_error
        and class_agreement
        and gate_agreement
    )
    payload = {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "onnx": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "execution_providers": session.get_providers(),
        "validation_window": {
            "dataset_root": str(args.dataset_root),
            "window_id": source["window_id"],
            "source_video": source["source_video"],
            "frame_paths": source["frame_paths"],
            "class_name": source["class_name"],
        },
        "input_shape": list(inputs.shape),
        "outputs": {
            "pytorch_logits": torch_logits_array.tolist(),
            "onnx_logits": onnx_logits.tolist(),
            "pytorch_gameplay_score": torch_score_array.tolist(),
            "onnx_gameplay_score": onnx_score.tolist(),
            "logit_difference": logit_difference,
            "score_difference": score_difference,
            "class_agreement": class_agreement,
            "gate_threshold": args.gate_threshold,
            "gate_decision_agreement": gate_agreement,
        },
        "tolerances": {
            "max_logit_error": args.max_logit_error,
            "max_score_error": args.max_score_error,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
