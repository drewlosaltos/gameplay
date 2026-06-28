#!/usr/bin/env python3
"""Build a TensorRT engine from the exported ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorrt as trt

from skyball_trt_common import default_artifact_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--workspace-gib", type=float, default=8.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = args.engine or (
        default_artifact_dir() / f"{args.onnx.stem}_{args.precision}.engine"
    )
    engine.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    network_flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with (
        trt.Builder(logger) as builder,
        builder.create_network(network_flags) as network,
        trt.OnnxParser(network, logger) as parser,
        builder.create_builder_config() as config,
    ):
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(args.workspace_gib * (1 << 30)),
        )
        if args.precision == "fp16" and hasattr(trt.BuilderFlag, "FP16"):
            config.set_flag(trt.BuilderFlag.FP16)
        elif args.precision == "bf16" and hasattr(trt.BuilderFlag, "BF16"):
            config.set_flag(trt.BuilderFlag.BF16)
        elif args.precision in {"fp16", "bf16"}:
            raise RuntimeError(
                f"TensorRT {trt.__version__} does not expose weak-typing "
                f"{args.precision} builder flags. Convert the ONNX graph to "
                "mixed precision first, then build with --precision fp32."
            )

        if not parser.parse(args.onnx.read_bytes()):
            print("ONNX parse failed:")
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise SystemExit(2)

        print("network inputs:")
        for idx in range(network.num_inputs):
            tensor = network.get_input(idx)
            print(f"  {tensor.name}: {tensor.shape} {tensor.dtype}")
        print("network outputs:")
        for idx in range(network.num_outputs):
            tensor = network.get_output(idx)
            print(f"  {tensor.name}: {tensor.shape} {tensor.dtype}")

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        engine.write_bytes(bytes(serialized))

    sidecar = engine.with_suffix(engine.suffix + ".json")
    write_json(
        sidecar,
        {
            "onnx": str(args.onnx),
            "engine": str(engine),
            "precision": args.precision,
            "workspace_gib": args.workspace_gib,
            "tensorrt_version": trt.__version__,
        },
    )
    print(f"wrote {engine}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
