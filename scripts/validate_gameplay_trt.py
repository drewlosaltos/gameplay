#!/usr/bin/env python3
"""Validate a gameplay TensorRT engine against PyTorch on a real window."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy
import pycuda.autoinit  # noqa: F401
import pycuda.driver
import tensorrt
import torch

REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyball_gameplay.modeling import load_checkpoint_model as load_model
from skyball_gameplay.modeling import sha256_file
from export_gameplay_onnx import DEFAULT_CHECKPOINT
from export_gameplay_onnx import GameplayExportWrapper
from validate_gameplay_onnx import DEFAULT_DATASET_ROOT
from validate_gameplay_onnx import difference_metrics
from validate_gameplay_onnx import load_validation_window


def run_engine(engine_path: Path, frames: numpy.ndarray) -> dict[str, numpy.ndarray]:
    """Run one gameplay batch through a serialized TensorRT engine.

    Args:
        engine_path: TensorRT engine built for the local runtime.
        frames: Normalized input array shaped ``[B, 4, 3, 512, 512]``.

    Returns:
        Output arrays keyed by TensorRT tensor name.
    """
    logger = tensorrt.Logger(tensorrt.Logger.ERROR)
    tensorrt.init_libnvinfer_plugins(logger, "")
    with tensorrt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if not context.set_input_shape("frames", tuple(frames.shape)):
        raise RuntimeError(f"TensorRT rejected input shape {frames.shape}")
    stream = pycuda.driver.Stream()
    allocations: list[Any] = []
    host_outputs: dict[str, numpy.ndarray] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        shape = tuple(context.get_tensor_shape(name))
        dtype = numpy.dtype(tensorrt.nptype(engine.get_tensor_dtype(name)))
        allocation = pycuda.driver.mem_alloc(int(numpy.prod(shape)) * dtype.itemsize)
        allocations.append(allocation)
        context.set_tensor_address(name, int(allocation))
        if engine.get_tensor_mode(name) == tensorrt.TensorIOMode.INPUT:
            pycuda.driver.memcpy_htod_async(
                allocation,
                numpy.ascontiguousarray(frames, dtype=dtype),
                stream,
            )
        else:
            host_outputs[name] = numpy.empty(shape, dtype=dtype)
    if not context.execute_async_v3(stream_handle=stream.handle):
        raise RuntimeError("TensorRT execute_async_v3 failed")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name in host_outputs:
            pycuda.driver.memcpy_dtoh_async(
                host_outputs[name],
                allocations[index],
                stream,
            )
    stream.synchronize()
    return host_outputs


def parse_args() -> argparse.Namespace:
    """Parse TensorRT parity arguments.

    Returns:
        Parsed parity-validation settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-threshold", type=float, default=0.2)
    parser.add_argument("--max-logit-error", type=float, default=2e-3)
    parser.add_argument("--max-score-error", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    """Execute real-window TensorRT parity validation and write evidence."""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing existing parity output: {args.output}")
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
    trt_outputs = run_engine(args.engine, inputs.numpy())
    trt_logits = trt_outputs["logits"]
    trt_score = trt_outputs["gameplay_score"]
    logit_difference = difference_metrics(torch_logits_array, trt_logits)
    score_difference = difference_metrics(torch_score_array, trt_score)
    class_agreement = bool(
        numpy.array_equal(torch_logits_array.argmax(axis=1), trt_logits.argmax(axis=1))
    )
    gate_agreement = bool(
        numpy.array_equal(
            torch_score_array >= args.gate_threshold,
            trt_score >= args.gate_threshold,
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
        "engine": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "tensorrt_version": tensorrt.__version__,
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
            "tensorrt_logits": trt_logits.tolist(),
            "pytorch_gameplay_score": torch_score_array.tolist(),
            "tensorrt_gameplay_score": trt_score.tolist(),
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
