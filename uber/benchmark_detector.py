#!/usr/bin/env python3
"""Benchmark PyTorch vs TensorRT for the SkyBall large detector."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorrt as trt
import torch

from skyball_trt_common import (
    DEFAULT_CHECKPOINT,
    autocast_context,
    default_artifact_dir,
    load_export_model,
    make_input,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--engine", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "bf16"),
        default="fp16",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-trt", action="store_true")
    return parser.parse_args()


def summarize_ms(samples_ms: List[float], batch_size: int = 1) -> Dict[str, float]:
    arr = np.asarray(samples_ms, dtype=np.float64)
    batch_fps = float(1000.0 / arr.mean())
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "batch_fps": batch_fps,
        "fps": float(batch_fps * batch_size),
    }


def benchmark_pytorch(args: argparse.Namespace) -> Dict[str, float]:
    device = torch.device(args.device)
    model, _ = load_export_model(
        args.checkpoint,
        device=device,
        output_mode="all-frames",
    )
    x = make_input(
        args.batch_size,
        args.num_frames,
        args.resolution,
        device=device,
        dtype=torch.float32,
    )
    samples = []
    with torch.inference_mode():
        with autocast_context(device, args.precision):
            for _ in range(args.warmup):
                _ = model(x)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with autocast_context(device, args.precision):
                _ = model(x)
            end.record()
            torch.cuda.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return summarize_ms(samples, args.batch_size)


class TrtRunner:
    def __init__(self, engine_path: Path):
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda

        self.cuda = cuda
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, "")
        with trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
        ]
        self.inputs = []
        self.outputs = []
        self.allocations = {}
        for name in self.names:
            shape = tuple(self.engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Dynamic tensor shapes are not supported here: {name} {shape}")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            ptr = cuda.mem_alloc(nbytes)
            self.allocations[name] = ptr
            self.context.set_tensor_address(name, int(ptr))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append((name, shape, dtype, ptr))
            else:
                self.outputs.append((name, shape, dtype, ptr))

    def seed_inputs(self) -> None:
        for _name, shape, dtype, ptr in self.inputs:
            host = np.random.randn(*shape).astype(dtype)
            self.cuda.memcpy_htod_async(ptr, host, self.stream)
        self.stream.synchronize()

    def run_once(self) -> None:
        ok = self.context.execute_async_v3(stream_handle=self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

    def benchmark(self, warmup: int, iters: int) -> Dict[str, float]:
        self.seed_inputs()
        for _ in range(warmup):
            self.run_once()
        self.stream.synchronize()

        samples = []
        for _ in range(iters):
            start = self.cuda.Event()
            end = self.cuda.Event()
            start.record(self.stream)
            self.run_once()
            end.record(self.stream)
            end.synchronize()
            samples.append(float(start.time_till(end)))
        batch_size = int(self.inputs[0][1][0])
        return summarize_ms(samples, batch_size)


def benchmark_trt(args: argparse.Namespace) -> Dict[str, float]:
    if args.engine is None:
        raise ValueError("--engine is required unless --skip-trt is set")
    runner = TrtRunner(args.engine)
    return runner.benchmark(args.warmup, args.iters)


def main() -> None:
    args = parse_args()
    payload = {
        "checkpoint": str(args.checkpoint),
        "engine": None if args.engine is None else str(args.engine),
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "resolution": args.resolution,
        "precision": args.precision,
        "output_mode": "all-frames",
        "warmup": args.warmup,
        "iters": args.iters,
    }
    if not args.skip_pytorch:
        payload["pytorch"] = benchmark_pytorch(args)
        print("PyTorch:", payload["pytorch"])
    if not args.skip_trt:
        payload["tensorrt"] = benchmark_trt(args)
        print("TensorRT:", payload["tensorrt"])

    output_json = args.output_json or (
        default_artifact_dir()
        / (
            f"benchmark_{args.precision}_allframes_b{args.batch_size}_"
            f"t{args.num_frames}_{args.resolution}.json"
        )
    )
    write_json(output_json, payload)
    print(f"wrote {output_json}")


if __name__ == "__main__":
    main()
