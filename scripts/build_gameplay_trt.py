#!/usr/bin/env python3
"""Build a machine-specific TensorRT engine from gameplay ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import tensorrt


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_description(tensor: Any) -> dict[str, Any]:
    """Serialize one TensorRT network tensor.

    Args:
        tensor: TensorRT input or output tensor.

    Returns:
        Tensor name, shape, and data type.
    """
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def parse_args() -> argparse.Namespace:
    """Parse TensorRT build arguments.

    Returns:
        Parsed build settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=4)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--disable-tf32", action="store_true",
                        help="Disable TF32 tactics for stricter FP32 parity.")
    return parser.parse_args()


def main() -> None:
    """Parse gameplay ONNX, build a dynamic-batch engine, and record evidence."""
    args = parse_args()
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)
    for output in (args.engine, args.evidence):
        if output.exists():
            raise FileExistsError(f"Refusing existing output: {output}")
    if not 0 < args.min_batch <= args.opt_batch <= args.max_batch:
        raise ValueError("Require 0 < min-batch <= opt-batch <= max-batch")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.parent.mkdir(parents=True, exist_ok=True)

    logger = tensorrt.Logger(tensorrt.Logger.INFO)
    tensorrt.init_libnvinfer_plugins(logger, "")
    network_flags = 0
    if hasattr(tensorrt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags |= 1 << int(
            tensorrt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH
        )
    with (
        tensorrt.Builder(logger) as builder,
        builder.create_network(network_flags) as network,
        tensorrt.OnnxParser(network, logger) as parser,
        builder.create_builder_config() as config,
    ):
        if not parser.parse(args.onnx.read_bytes()):
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
        if network.num_inputs != 1 or network.get_input(0).name != "frames":
            raise ValueError("Expected one ONNX input named frames")
        frames = network.get_input(0)
        trailing_shape = tuple(int(value) for value in frames.shape[1:])
        if trailing_shape != (4, 3, 512, 512):
            raise ValueError(f"Unexpected gameplay input shape: {frames.shape}")
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "frames",
            (args.min_batch, *trailing_shape),
            (args.opt_batch, *trailing_shape),
            (args.max_batch, *trailing_shape),
        )
        config.add_optimization_profile(profile)
        if args.disable_tf32:
            config.clear_flag(tensorrt.BuilderFlag.TF32)
        config.set_memory_pool_limit(
            tensorrt.MemoryPoolType.WORKSPACE,
            int(args.workspace_gib * (1 << 30)),
        )
        inputs = [
            tensor_description(network.get_input(index))
            for index in range(network.num_inputs)
        ]
        outputs = [
            tensor_description(network.get_output(index))
            for index in range(network.num_outputs)
        ]
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        args.engine.write_bytes(bytes(serialized))

    payload = {
        "schema_version": 1,
        "status": "pass",
        "tensorrt_version": tensorrt.__version__,
        "onnx": str(args.onnx),
        "onnx_sha256": sha256_file(args.onnx),
        "engine": str(args.engine),
        "engine_sha256": sha256_file(args.engine),
        "engine_size_bytes": args.engine.stat().st_size,
        "precision": "fp32",
        "tf32_enabled": not args.disable_tf32,
        "workspace_gib": args.workspace_gib,
        "optimization_profile": {
            "min_batch": args.min_batch,
            "opt_batch": args.opt_batch,
            "max_batch": args.max_batch,
        },
        "network_inputs": inputs,
        "network_outputs": outputs,
        "publication_note": (
            "The engine is local machine-specific validation evidence and is not "
            "a portable published artifact; consumers build from the checksum-pinned ONNX."
        ),
    }
    args.evidence.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
