#!/usr/bin/env python3
"""Convert an ONNX graph to mixed precision for TensorRT 11 strongly typed builds."""

from __future__ import annotations

import argparse
from pathlib import Path

import modelopt.onnx.autocast as autocast
import onnx

from skyball_trt_common import default_artifact_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--keep-io-types", action="store_true", default=True)
    parser.add_argument(
        "--op-types-to-exclude",
        nargs="*",
        default=None,
        help="ONNX op types to keep in FP32.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (
        default_artifact_dir() / f"{args.onnx.stem}_{args.precision}.onnx"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    model = autocast.convert_to_mixed_precision(
        onnx_path=str(args.onnx),
        low_precision_type=args.precision,
        keep_io_types=args.keep_io_types,
        op_types_to_exclude=args.op_types_to_exclude,
        providers=["cpu"],
    )
    onnx.save(model, output)
    sidecar = output.with_suffix(output.suffix + ".json")
    write_json(
        sidecar,
        {
            "source_onnx": str(args.onnx),
            "output_onnx": str(output),
            "precision": args.precision,
            "keep_io_types": args.keep_io_types,
            "op_types_to_exclude": args.op_types_to_exclude,
        },
    )
    print(f"wrote {output}")
    print(f"wrote {sidecar}")


if __name__ == "__main__":
    main()
