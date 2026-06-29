#!/usr/bin/env python3
"""Compare PyTorch and TensorRT outputs on real SkyBall video windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch

from skyball_trt_common import DEFAULT_CHECKPOINT


DETECTOR_REPO = Path("/home/atao/vsdevel/skyball/rf-detr-mf-skyball")
DEFAULT_ENGINE = Path(
    "/mnt/t/output/rf-detr-mf/benchmarks/"
    "large_dense_v4_addcats_sparse_v3_2g_trt/"
    "large_addcats_b1_t6_1024_allframes_trt11_fp16.engine"
)
DEFAULT_VIDEO = Path(
    "/mnt/t/data/vball/skyball/unified_dataset/video/"
    "000004_intrepid_iguana_play1.mp4"
)


def load_predict_tracks_module():
    path = DETECTOR_REPO / "scripts" / "predict_tracks.py"
    spec = importlib.util.spec_from_file_location("predict_tracks_diag", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    a = a.detach().cpu().float()
    b = b.detach().cpu().float()
    diff = (a - b).abs()
    return {
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
        "p95_abs": float(torch.quantile(diff.flatten(), 0.95).item()),
        "p99_abs": float(torch.quantile(diff.flatten(), 0.99).item()),
        "a_min": float(a.min().item()),
        "a_max": float(a.max().item()),
        "b_min": float(b.min().item()),
        "b_max": float(b.max().item()),
    }


def class_counts(logits: torch.Tensor, threshold: float) -> Dict[int, int]:
    scores, labels = logits.detach().cpu().float().sigmoid().max(dim=-1)
    keep = scores >= threshold
    counts: Dict[int, int] = {}
    for label in labels[keep].flatten().tolist():
        counts[int(label)] = counts.get(int(label), 0) + 1
    return dict(sorted(counts.items()))


def add_counts(dst: Dict[int, int], src: Dict[int, int]) -> None:
    for key, value in src.items():
        dst[int(key)] = dst.get(int(key), 0) + int(value)


def serial_counts(counts: Dict[int, int]) -> Dict[str, int]:
    return {str(key): counts[key] for key in sorted(counts)}


def nonfinite_by_class(logits: torch.Tensor) -> Dict[int, int]:
    logits = logits.detach().cpu().float()
    finite = torch.isfinite(logits)
    bad_pos = ~finite.all(dim=-1)
    if not bad_pos.any():
        return {}
    safe_logits = logits.clone()
    safe_logits[torch.isnan(safe_logits)] = -torch.inf
    labels = safe_logits.argmax(dim=-1)
    counts: Dict[int, int] = {}
    for label in labels[bad_pos].flatten().tolist():
        counts[int(label)] = counts.get(int(label), 0) + 1
    return dict(sorted(counts.items()))


def referee_summary(logits: torch.Tensor) -> Dict[str, float]:
    probs = logits.detach().cpu().float().sigmoid()
    ref = probs[..., 5]
    scores, labels = probs.max(dim=-1)
    ref_wins = labels == 5
    return {
        "ref_prob_mean": float(ref.mean().item()),
        "ref_prob_max": float(ref.max().item()),
        "ref_winner_count_050": int(((scores >= 0.50) & ref_wins).sum().item()),
        "ref_winner_count_080": int(((scores >= 0.80) & ref_wins).sum().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--video-max-frames", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    predict_tracks = load_predict_tracks_module()
    from rfdetr.util.tracking_model_loader import load_model_from_checkpoint

    device = torch.device("cuda")
    torch_model, saved_args = load_model_from_checkpoint(
        str(args.checkpoint),
        device,
        checkpoint_state="model",
    )
    torch_model.eval()
    resolution = int(getattr(saved_args, "resolution", 1024))
    window_size = int(getattr(saved_args, "num_frames", 6))

    frames, _fps, _shape = predict_tracks.read_video(
        str(args.video),
        max_frames=args.video_max_frames,
    )
    trt_model = predict_tracks.TensorRTTrackingRunner(str(args.engine))
    trt_batch_size = int(trt_model.input_shape[0])
    if trt_batch_size < 1:
        raise RuntimeError(f"Invalid TensorRT batch size: {trt_batch_size}")

    keys = [
        "pred_boxes",
        "pred_logits",
        "pred_boxes_all_frames",
        "pred_logits_all_frames",
        "pred_frame_event_logits",
        "pred_frame_skill_logits",
        "pred_frame_side_logits",
        "pred_frame_event_occ_logits",
        "pred_frame_actor_logits",
        "pred_side_logits",
        "pred_jump_logits_all_frames",
    ]
    starts = list(range(args.window_start, len(frames) - window_size + 1, args.stride))
    if args.max_windows > 0:
        starts = starts[: args.max_windows]
    if not starts:
        raise RuntimeError("No windows selected")
    payload = {
        "checkpoint": str(args.checkpoint),
        "engine": str(args.engine),
        "video": str(args.video),
        "video_max_frames": args.video_max_frames,
        "window_start": args.window_start,
        "stride": args.stride,
        "windows": len(starts),
        "trt_batch_size": trt_batch_size,
        "threshold": args.threshold,
        "diffs": {},
        "class_counts": {},
        "referee": {},
        "nonfinite": {},
    }
    diff_values: Dict[str, list[Dict[str, float]]] = {key: [] for key in keys}
    count_totals = {
        "pred_logits": {"pytorch": {}, "tensorrt": {}},
        "pred_logits_all_frames": {"pytorch": {}, "tensorrt": {}},
    }
    referee_totals = {
        "pred_logits": {
            "pytorch_ref_winner_count_050": 0,
            "pytorch_ref_winner_count_080": 0,
            "tensorrt_ref_winner_count_050": 0,
            "tensorrt_ref_winner_count_080": 0,
        },
        "pred_logits_all_frames": {
            "pytorch_ref_winner_count_050": 0,
            "pytorch_ref_winner_count_080": 0,
            "tensorrt_ref_winner_count_050": 0,
            "tensorrt_ref_winner_count_080": 0,
        },
    }
    nonfinite_totals = {key: {"pytorch": {}, "tensorrt": {}} for key in keys}

    with torch.inference_mode():
        for batch_start in range(0, len(starts), trt_batch_size):
            batch_starts = starts[batch_start : batch_start + trt_batch_size]
            batch_windows = [
                frames[start : start + window_size]
                for start in batch_starts
            ]
            padded_windows = list(batch_windows)
            while len(padded_windows) < trt_batch_size:
                padded_windows.append(batch_windows[-1])
            torch_samples = predict_tracks.preprocess_window_batch(
                padded_windows,
                resolution,
                device,
            )
            trt_samples = predict_tracks.preprocess_window_batch_array(
                padded_windows,
                resolution,
            )
            torch_outputs = torch_model(torch_samples)
            trt_outputs = trt_model.run_array(
                trt_samples,
                return_numpy=False,
                copy_outputs=True,
            )

            for key in keys:
                if key not in torch_outputs or key not in trt_outputs:
                    continue
                for batch_idx in range(len(batch_windows)):
                    torch_value = torch_outputs[key][batch_idx : batch_idx + 1]
                    trt_value = trt_outputs[key][batch_idx : batch_idx + 1]
                    diff_values[key].append(tensor_stats(torch_value, trt_value))
                    add_counts(
                        nonfinite_totals[key]["pytorch"],
                        nonfinite_by_class(torch_value),
                    )
                    add_counts(
                        nonfinite_totals[key]["tensorrt"],
                        nonfinite_by_class(trt_value),
                    )

            for key in ["pred_logits", "pred_logits_all_frames"]:
                for batch_idx in range(len(batch_windows)):
                    torch_value = torch_outputs[key][batch_idx : batch_idx + 1]
                    trt_value = trt_outputs[key][batch_idx : batch_idx + 1]
                    add_counts(
                        count_totals[key]["pytorch"],
                        class_counts(torch_value, args.threshold),
                    )
                    add_counts(
                        count_totals[key]["tensorrt"],
                        class_counts(trt_value, args.threshold),
                    )
                    pt_ref = referee_summary(torch_value)
                    trt_ref = referee_summary(trt_value)
                    referee_totals[key]["pytorch_ref_winner_count_050"] += int(
                        pt_ref["ref_winner_count_050"]
                    )
                    referee_totals[key]["pytorch_ref_winner_count_080"] += int(
                        pt_ref["ref_winner_count_080"]
                    )
                    referee_totals[key]["tensorrt_ref_winner_count_050"] += int(
                        trt_ref["ref_winner_count_050"]
                    )
                    referee_totals[key]["tensorrt_ref_winner_count_080"] += int(
                        trt_ref["ref_winner_count_080"]
                    )

    for key in keys:
        stats = diff_values.get(key, [])
        if not stats:
            continue
        payload["diffs"][key] = {
            "mean_abs_avg": float(sum(v["mean_abs"] for v in stats) / len(stats)),
            "max_abs_max": float(max(v["max_abs"] for v in stats)),
            "p95_abs_avg": float(sum(v["p95_abs"] for v in stats) / len(stats)),
            "p99_abs_avg": float(sum(v["p99_abs"] for v in stats) / len(stats)),
        }
        payload["nonfinite"][key] = {
            "pytorch": serial_counts(nonfinite_totals[key]["pytorch"]),
            "tensorrt": serial_counts(nonfinite_totals[key]["tensorrt"]),
        }

    for key in ["pred_logits", "pred_logits_all_frames"]:
        payload["class_counts"][key] = {
            "pytorch": serial_counts(count_totals[key]["pytorch"]),
            "tensorrt": serial_counts(count_totals[key]["tensorrt"]),
        }
        payload["referee"][key] = referee_totals[key]

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n")


if __name__ == "__main__":
    main()
