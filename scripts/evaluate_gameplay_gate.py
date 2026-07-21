#!/usr/bin/env python3
"""Score full match videos and evaluate conservative gameplay gating.

This is a standalone feasibility tool. It does not integrate with the SkyBall
pipeline. The scorer evaluates the registered four-frame gameplay classifier at
one-second steps while encoding each sampled frame only once. The analyzer turns
window scores into coarse intervals and measures coverage against synchronized
rally annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy
import torch
import torch.nn.functional
import yaml

REPO_ROOT = Path(__file__).parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyball_gameplay.modeling import TemporalAveragePoolClassifier
from skyball_gameplay.modeling import TemporalFeatureAveragePoolClassifier
from skyball_gameplay.modeling import load_checkpoint_model
from skyball_gameplay.modeling import sha256_file


REGISTRY_MODEL = yaml.safe_load(
    (REPO_ROOT / "model_registry.yaml").read_text(encoding="utf-8")
)["best_model"]
DEFAULT_EVALUATION_ROOT = Path("/mnt/t/data/vball/skyball/evaluation")
DEFAULT_OUTPUT_ROOT = (
    Path("/mnt/t/output/skyball_gameplay")
    / f"gameplay_gate_eval10_{REGISTRY_MODEL['model_id']}"
)
DEFAULT_CHECKPOINT = Path(REGISTRY_MODEL["checkpoint"])
DEFAULT_REGISTRY_THRESHOLD = float(REGISTRY_MODEL["threshold"])


@dataclass(frozen=True, order=True)
class Interval:
    """A half-open interval in video seconds."""

    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        """Validate interval ordering and finiteness."""
        if not math.isfinite(self.start_sec) or not math.isfinite(self.end_sec):
            raise ValueError(f"Interval endpoints must be finite: {self}")
        if self.end_sec < self.start_sec:
            raise ValueError(f"Interval end precedes start: {self}")


@dataclass(frozen=True)
class GroundTruthRally:
    """One synchronized rally and its confirmed event times."""

    set_number: str
    rally_number: str
    interval: Interval
    event_times_sec: tuple[float, ...]


@dataclass(frozen=True)
class Policy:
    """Temporal postprocessing parameters for gameplay scores."""

    threshold: float
    merge_gap_sec: float
    pad_before_sec: float
    pad_after_sec: float

    @property
    def name(self) -> str:
        """Return a stable compact policy identifier."""
        return (
            f"thr{self.threshold:.9g}_gap{self.merge_gap_sec:g}_"
            f"pre{self.pad_before_sec:g}_post{self.pad_after_sec:g}"
        )


@dataclass(frozen=True)
class MatchData:
    """Cached scores, annotations, and duration for one match."""

    match_id: str
    starts_sec: tuple[float, ...]
    scores: tuple[float, ...]
    window_sec: float
    duration_sec: float
    rallies: tuple[GroundTruthRally, ...]


def merge_intervals(
    intervals: Sequence[Interval],
    max_gap_sec: float = 0.0,
) -> list[Interval]:
    """Return sorted interval union, optionally filling short gaps."""
    if max_gap_sec < 0:
        raise ValueError("max_gap_sec must be nonnegative")
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.start_sec <= previous.end_sec + max_gap_sec:
            merged[-1] = Interval(
                previous.start_sec,
                max(previous.end_sec, interval.end_sec),
            )
        else:
            merged.append(interval)
    return merged


def predicted_intervals(
    starts_sec: Sequence[float],
    scores: Sequence[float],
    threshold: float,
    window_sec: float,
    merge_gap_sec: float,
    pad_before_sec: float,
    pad_after_sec: float,
    video_duration_sec: float,
) -> list[Interval]:
    """Convert thresholded window scores into padded coarse video intervals."""
    if len(starts_sec) != len(scores):
        raise ValueError("starts_sec and scores must have equal length")
    if window_sec <= 0 or video_duration_sec < 0:
        raise ValueError("window_sec must be positive and duration nonnegative")
    if pad_before_sec < 0 or pad_after_sec < 0:
        raise ValueError("padding must be nonnegative")
    positive = [
        Interval(float(start), min(video_duration_sec, float(start) + window_sec))
        for start, score in zip(starts_sec, scores, strict=True)
        if float(score) >= threshold and float(start) < video_duration_sec
    ]
    merged = merge_intervals(positive, max_gap_sec=merge_gap_sec)
    padded = [
        Interval(
            max(0.0, interval.start_sec - pad_before_sec),
            min(video_duration_sec, interval.end_sec + pad_after_sec),
        )
        for interval in merged
    ]
    return merge_intervals(padded)


def interval_union_duration(intervals: Sequence[Interval]) -> float:
    """Return union duration without double-counting overlap."""
    return sum(interval.end_sec - interval.start_sec for interval in merge_intervals(intervals))


def interval_intersection_duration(
    left: Sequence[Interval],
    right: Sequence[Interval],
) -> float:
    """Return the duration of the intersection of two interval unions."""
    left_merged = merge_intervals(left)
    right_merged = merge_intervals(right)
    left_index = 0
    right_index = 0
    total = 0.0
    while left_index < len(left_merged) and right_index < len(right_merged):
        left_interval = left_merged[left_index]
        right_interval = right_merged[right_index]
        total += max(
            0.0,
            min(left_interval.end_sec, right_interval.end_sec)
            - max(left_interval.start_sec, right_interval.start_sec),
        )
        if left_interval.end_sec <= right_interval.end_sec:
            left_index += 1
        else:
            right_index += 1
    return total


def interval_contains(intervals: Sequence[Interval], target: Interval) -> bool:
    """Return whether an interval union completely contains a target interval."""
    tolerance = 1e-9
    return any(
        interval.start_sec <= target.start_sec + tolerance
        and interval.end_sec + tolerance >= target.end_sec
        for interval in merge_intervals(intervals)
    )


def point_is_covered(intervals: Sequence[Interval], point_sec: float) -> bool:
    """Return whether a timestamp lies within the interval union."""
    tolerance = 1e-9
    return any(
        interval.start_sec - tolerance <= point_sec <= interval.end_sec + tolerance
        for interval in intervals
    )


def coverage_metrics(
    predicted: Sequence[Interval],
    rallies: Sequence[GroundTruthRally],
    video_duration_sec: float,
) -> dict[str, float | int]:
    """Measure conservative gameplay coverage and retained video time."""
    gt_intervals = merge_intervals([rally.interval for rally in rallies])
    predicted_merged = merge_intervals(predicted)
    gameplay_seconds = interval_union_duration(gt_intervals)
    retained_seconds = interval_union_duration(predicted_merged)
    covered_seconds = interval_intersection_duration(predicted_merged, gt_intervals)
    events = [time_sec for rally in rallies for time_sec in rally.event_times_sec]
    covered_events = sum(point_is_covered(predicted_merged, event) for event in events)
    fully_covered_rallies = sum(
        interval_contains(predicted_merged, rally.interval) for rally in rallies
    )
    negative_seconds = max(0.0, video_duration_sec - gameplay_seconds)
    retained_negative_seconds = max(0.0, retained_seconds - covered_seconds)
    return {
        "video_duration_seconds": video_duration_sec,
        "annotated_gameplay_seconds": gameplay_seconds,
        "covered_gameplay_seconds": covered_seconds,
        "missed_gameplay_seconds": max(0.0, gameplay_seconds - covered_seconds),
        "annotated_gameplay_recall": (
            covered_seconds / gameplay_seconds if gameplay_seconds else 0.0
        ),
        "events": len(events),
        "covered_events": covered_events,
        "event_recall": covered_events / len(events) if events else 0.0,
        "rallies": len(rallies),
        "fully_covered_rallies": fully_covered_rallies,
        "rally_full_coverage_recall": (
            fully_covered_rallies / len(rallies) if rallies else 0.0
        ),
        "retained_total_seconds": retained_seconds,
        "retained_video_fraction": (
            retained_seconds / video_duration_sec if video_duration_sec else 0.0
        ),
        "excluded_video_seconds": max(0.0, video_duration_sec - retained_seconds),
        "excluded_video_fraction": (
            max(0.0, video_duration_sec - retained_seconds) / video_duration_sec
            if video_duration_sec
            else 0.0
        ),
        "annotation_negative_seconds": negative_seconds,
        "retained_annotation_negative_seconds": retained_negative_seconds,
        "retained_annotation_negative_fraction": (
            retained_negative_seconds / negative_seconds if negative_seconds else 0.0
        ),
        "predicted_intervals": len(predicted_merged),
    }


def load_ground_truth_rallies(
    path: Path,
    margin_sec: float,
    video_duration_sec: float,
) -> tuple[GroundTruthRally, ...]:
    """Load confirmed events and group them into margin-expanded rallies."""
    if margin_sec < 0:
        raise ValueError("margin_sec must be nonnegative")
    grouped: dict[tuple[str, str], list[float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("status") or "").strip().lower() != "confirmed":
                continue
            raw_time = str(row.get("synced_sec") or "").strip()
            if not raw_time:
                continue
            key = (str(row.get("set") or ""), str(row.get("rally") or ""))
            grouped.setdefault(key, []).append(float(raw_time))
    rallies = []
    for (set_number, rally_number), times_sec in grouped.items():
        ordered_times = tuple(sorted(times_sec))
        rallies.append(
            GroundTruthRally(
                set_number=set_number,
                rally_number=rally_number,
                interval=Interval(
                    max(0.0, ordered_times[0] - margin_sec),
                    min(video_duration_sec, ordered_times[-1] + margin_sec),
                ),
                event_times_sec=ordered_times,
            )
        )
    return tuple(sorted(rallies, key=lambda rally: rally.interval.start_sec))


def aggregate_metrics(rows: Iterable[dict[str, float | int]]) -> dict[str, float | int]:
    """Aggregate per-match coverage counts and durations."""
    rows_list = list(rows)
    sum_keys = (
        "video_duration_seconds",
        "annotated_gameplay_seconds",
        "covered_gameplay_seconds",
        "missed_gameplay_seconds",
        "events",
        "covered_events",
        "rallies",
        "fully_covered_rallies",
        "retained_total_seconds",
        "excluded_video_seconds",
        "annotation_negative_seconds",
        "retained_annotation_negative_seconds",
        "predicted_intervals",
    )
    result: dict[str, float | int] = {
        key: sum(float(row[key]) for row in rows_list) for key in sum_keys
    }
    for integer_key in (
        "events",
        "covered_events",
        "rallies",
        "fully_covered_rallies",
        "predicted_intervals",
    ):
        result[integer_key] = int(round(float(result[integer_key])))
    gameplay = float(result["annotated_gameplay_seconds"])
    covered = float(result["covered_gameplay_seconds"])
    duration = float(result["video_duration_seconds"])
    retained = float(result["retained_total_seconds"])
    negative = float(result["annotation_negative_seconds"])
    retained_negative = float(result["retained_annotation_negative_seconds"])
    events = int(result["events"])
    rallies = int(result["rallies"])
    result.update(
        {
            "annotated_gameplay_recall": covered / gameplay if gameplay else 0.0,
            "event_recall": int(result["covered_events"]) / events if events else 0.0,
            "rally_full_coverage_recall": (
                int(result["fully_covered_rallies"]) / rallies if rallies else 0.0
            ),
            "retained_video_fraction": retained / duration if duration else 0.0,
            "excluded_video_fraction": (
                float(result["excluded_video_seconds"]) / duration if duration else 0.0
            ),
            "retained_annotation_negative_fraction": (
                retained_negative / negative if negative else 0.0
            ),
        }
    )
    return result


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the registered temporal classifier and checkpoint metadata."""
    return load_checkpoint_model(
        checkpoint_path,
        device,
        require_input_frames=4,
    )


def probe_video(path: Path) -> dict[str, float | int]:
    """Read video dimensions and duration with ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "duration_sec": float(payload["format"]["duration"]),
    }


def scaled_dimensions(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Return even dimensions after preserving aspect ratio at a long edge."""
    if width <= 0 or height <= 0 or long_edge <= 0:
        raise ValueError("dimensions and long_edge must be positive")
    scale = long_edge / max(width, height)
    scaled_width = max(2, int(round(width * scale / 2.0)) * 2)
    scaled_height = max(2, int(round(height * scale / 2.0)) * 2)
    return scaled_width, scaled_height


def read_exact(handle: BinaryIO, size: int) -> bytes:
    """Read up to exactly size bytes, retaining partial pipe reads."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = handle.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_video_frames(
    video_path: Path,
    model: torch.nn.Module,
    device: torch.device,
    image_size: Sequence[int],
    normalization_mean: Sequence[float],
    normalization_std: Sequence[float],
    sample_fps: float,
    frame_batch_size: int,
    decode_long_edge: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Decode sampled frames and return one encoder embedding per frame."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    video_info = probe_video(video_path)
    decode_width, decode_height = scaled_dimensions(
        int(video_info["width"]), int(video_info["height"]), decode_long_edge
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        (
            f"fps=fps={sample_fps}:round=near,"
            f"scale={decode_width}:{decode_height}:flags=bilinear"
        ),
        "-pix_fmt",
        "rgb24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.stdout is None or process.stderr is None:
        raise RuntimeError("ffmpeg pipes were not created")
    bytes_per_frame = decode_width * decode_height * 3
    mean = torch.tensor(normalization_mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(normalization_std, device=device).view(1, 3, 1, 1)
    embeddings = []
    frame_payloads: list[bytes] = []
    frame_count = 0
    start_time = time.perf_counter()

    def flush_batch() -> None:
        """Encode the currently buffered raw RGB frames."""
        nonlocal frame_count
        if not frame_payloads:
            return
        array = numpy.stack(
            [
                numpy.frombuffer(payload, dtype=numpy.uint8)
                .reshape(decode_height, decode_width, 3)
                .copy()
                for payload in frame_payloads
            ]
        )
        batch = torch.from_numpy(array).permute(0, 3, 1, 2).to(device=device)
        batch = batch.to(dtype=torch.float32).div_(255.0)
        batch = torch.nn.functional.interpolate(
            batch,
            size=(int(image_size[0]), int(image_size[1])),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        batch = (batch - mean) / std
        with torch.inference_mode():
            encoded = torch.flatten(model.encoder(batch), 1)
        embeddings.append(encoded.cpu())
        frame_count += len(frame_payloads)
        frame_payloads.clear()

    try:
        while True:
            payload = read_exact(process.stdout, bytes_per_frame)
            if not payload:
                break
            if len(payload) != bytes_per_frame:
                raise RuntimeError(
                    f"Truncated ffmpeg frame for {video_path}: "
                    f"{len(payload)} of {bytes_per_frame} bytes"
                )
            frame_payloads.append(payload)
            if len(frame_payloads) >= frame_batch_size:
                flush_batch()
        flush_batch()
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr.strip()}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
    elapsed = time.perf_counter() - start_time
    if not embeddings:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return torch.cat(embeddings, dim=0), {
        **video_info,
        "decoded_width": decode_width,
        "decoded_height": decode_height,
        "sampled_frames": frame_count,
        "elapsed_seconds": elapsed,
        "sampled_frames_per_wallclock_second": frame_count / elapsed if elapsed else 0.0,
    }


def score_embeddings(
    embeddings: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
    input_frames: int,
    gameplay_class_index: int,
    batch_size: int,
) -> list[float]:
    """Classify all sliding temporal windows from cached frame embeddings."""
    if embeddings.shape[0] < input_frames:
        return []
    window_embeddings = embeddings.unfold(0, input_frames, 1).mean(dim=-1)
    scores = []
    with torch.inference_mode():
        for start in range(0, window_embeddings.shape[0], batch_size):
            batch = window_embeddings[start : start + batch_size].to(device=device)
            logits = model.classifier(model.dropout(batch))
            probabilities = torch.softmax(logits, dim=1)[:, gameplay_class_index]
            scores.extend(float(value) for value in probabilities.cpu().tolist())
    return scores


def atomic_write_json(path: Path, payload: Any) -> None:
    """Write JSON through a sibling temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_write_scores(
    path: Path,
    starts_sec: Sequence[float],
    scores: Sequence[float],
    window_sec: float,
) -> None:
    """Write raw gameplay scores through a sibling temporary file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("window_start_sec", "window_end_sec", "center_sec", "score_gameplay"),
        )
        writer.writeheader()
        for start_sec, score in zip(starts_sec, scores, strict=True):
            writer.writerow(
                {
                    "window_start_sec": f"{start_sec:.6f}",
                    "window_end_sec": f"{start_sec + window_sec:.6f}",
                    "center_sec": f"{start_sec + window_sec / 2.0:.6f}",
                    "score_gameplay": f"{score:.10g}",
                }
            )
    temporary.replace(path)


def discover_matches(evaluation_root: Path) -> list[tuple[str, Path, Path]]:
    """Discover evaluation videos and synchronized annotations."""
    matches = []
    for match_dir in sorted(evaluation_root.glob("eval_*")):
        if not match_dir.is_dir():
            continue
        match_id = match_dir.name
        video = match_dir / f"{match_id}.mp4"
        ground_truth = match_dir / f"{match_id}_gt_synced.csv"
        if video.is_file() and ground_truth.is_file():
            matches.append((match_id, video, ground_truth))
    if not matches:
        raise FileNotFoundError(f"No evaluation matches found under {evaluation_root}")
    return matches


def score_command(args: argparse.Namespace) -> None:
    """Run resumable raw-video scoring for one deterministic match shard."""
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be within [0, num-shards)")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    model, model_metadata = load_model(args.checkpoint, device)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    matches = discover_matches(args.evaluation_root)
    shard = matches[args.shard_index :: args.num_shards]
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Scoring {len(shard)} of {len(matches)} matches on {device}; "
        f"shard {args.shard_index}/{args.num_shards}",
        flush=True,
    )
    for match_id, video_path, ground_truth_path in shard:
        match_output = args.output_root / match_id
        scores_path = match_output / "scores.csv"
        metadata_path = match_output / "score_metadata.json"
        if metadata_path.is_file() and scores_path.is_file() and not args.force:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if existing.get("checkpoint_sha256") != checkpoint_sha256:
                raise RuntimeError(
                    f"Existing score checkpoint mismatch for {match_id}; use a new output root"
                )
            print(f"Skipping completed {match_id}", flush=True)
            continue
        if (metadata_path.exists() or scores_path.exists()) and not args.force:
            raise RuntimeError(
                f"Partial output exists for {match_id}; inspect it or rerun with --force"
            )
        match_output.mkdir(parents=True, exist_ok=True)
        print(f"Scoring {match_id}: {video_path}", flush=True)
        embeddings, timing = encode_video_frames(
            video_path=video_path,
            model=model,
            device=device,
            image_size=model_metadata["image_size"],
            normalization_mean=model_metadata["normalization_mean"],
            normalization_std=model_metadata["normalization_std"],
            sample_fps=args.sample_fps,
            frame_batch_size=args.frame_batch_size,
            decode_long_edge=args.decode_long_edge,
        )
        score_start = time.perf_counter()
        scores = score_embeddings(
            embeddings=embeddings,
            model=model,
            device=device,
            input_frames=int(model_metadata["input_frames"]),
            gameplay_class_index=int(model_metadata["gameplay_class_index"]),
            batch_size=args.classifier_batch_size,
        )
        classifier_elapsed = time.perf_counter() - score_start
        starts_sec = [index / args.sample_fps for index in range(len(scores))]
        window_sec = int(model_metadata["input_frames"]) / args.sample_fps
        atomic_write_scores(scores_path, starts_sec, scores, window_sec)
        metadata = {
            "schema_version": 1,
            "standalone_feasibility_study": True,
            "match_id": match_id,
            "video": str(video_path),
            "ground_truth": str(ground_truth_path),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "device": str(device),
            "sample_fps": args.sample_fps,
            "window_sec": window_sec,
            "score_count": len(scores),
            "model": model_metadata,
            "preprocessing": {
                "decode_long_edge": args.decode_long_edge,
                "ffmpeg_scale": "bilinear_preserve_aspect",
                "model_resize": model_metadata["image_size"],
                "model_resize_interpolation": "torch_bilinear_antialias",
                "normalization_mean": model_metadata["normalization_mean"],
                "normalization_std": model_metadata["normalization_std"],
            },
            "timing": {
                **timing,
                "classifier_seconds": classifier_elapsed,
                "total_seconds": float(timing["elapsed_seconds"]) + classifier_elapsed,
            },
        }
        atomic_write_json(metadata_path, metadata)
        print(
            f"Completed {match_id}: {len(scores)} windows in "
            f"{metadata['timing']['total_seconds']:.1f}s",
            flush=True,
        )


def load_match_data(
    evaluation_root: Path,
    output_root: Path,
    gt_margin_sec: float,
) -> list[MatchData]:
    """Load completed scores and synchronized rally ground truth."""
    matches = []
    for match_id, _video_path, ground_truth_path in discover_matches(evaluation_root):
        match_output = output_root / match_id
        metadata_path = match_output / "score_metadata.json"
        scores_path = match_output / "scores.csv"
        if not metadata_path.is_file() or not scores_path.is_file():
            raise FileNotFoundError(f"Incomplete scoring output for {match_id}: {match_output}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        starts = []
        scores = []
        with scores_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                starts.append(float(row["window_start_sec"]))
                scores.append(float(row["score_gameplay"]))
        duration_sec = float(metadata["timing"]["duration_sec"])
        matches.append(
            MatchData(
                match_id=match_id,
                starts_sec=tuple(starts),
                scores=tuple(scores),
                window_sec=float(metadata["window_sec"]),
                duration_sec=duration_sec,
                rallies=load_ground_truth_rallies(
                    ground_truth_path,
                    margin_sec=gt_margin_sec,
                    video_duration_sec=duration_sec,
                ),
            )
        )
    return matches


def evaluate_policy(match: MatchData, policy: Policy) -> dict[str, float | int]:
    """Evaluate one temporal policy on one match."""
    predicted = predicted_intervals(
        starts_sec=match.starts_sec,
        scores=match.scores,
        threshold=policy.threshold,
        window_sec=match.window_sec,
        merge_gap_sec=policy.merge_gap_sec,
        pad_before_sec=policy.pad_before_sec,
        pad_after_sec=policy.pad_after_sec,
        video_duration_sec=match.duration_sec,
    )
    return coverage_metrics(predicted, match.rallies, match.duration_sec)


def candidate_policies(registry_threshold: float) -> list[Policy]:
    """Return a bounded conservative policy grid for the feasibility study."""
    thresholds = sorted(
        {
            0.0,
            0.001,
            0.002,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.075,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            registry_threshold,
            0.4,
            0.5,
        }
    )
    gaps = (0.0, 2.0, 5.0, 10.0, 20.0, 30.0)
    pads = (0.0, 2.0, 3.0, 5.0, 10.0, 20.0)
    return [
        Policy(threshold, gap, padding, padding)
        for threshold in thresholds
        for gap in gaps
        for padding in pads
    ]


def is_full_recall(metrics: dict[str, float | int]) -> bool:
    """Return whether every coverage-oriented recall is exactly complete."""
    tolerance = 1e-9
    return (
        float(metrics["annotated_gameplay_recall"]) >= 1.0 - tolerance
        and float(metrics["event_recall"]) >= 1.0 - tolerance
        and float(metrics["rally_full_coverage_recall"]) >= 1.0 - tolerance
    )


def select_most_selective_full_recall(
    policy_metrics: Iterable[tuple[Policy, dict[str, float | int]]],
) -> tuple[Policy, dict[str, float | int]]:
    """Select the least-retaining policy among policies with complete recall."""
    eligible = [(policy, metrics) for policy, metrics in policy_metrics if is_full_recall(metrics)]
    if not eligible:
        raise RuntimeError("No full-recall policy was evaluated")
    return min(
        eligible,
        key=lambda item: (
            float(item[1]["retained_total_seconds"]),
            float(item[1]["retained_annotation_negative_seconds"]),
            -item[0].threshold,
            item[0].pad_before_sec + item[0].pad_after_sec,
            item[0].merge_gap_sec,
        ),
    )


def analyze_command(args: argparse.Namespace) -> None:
    """Sweep temporal policies and write aggregate feasibility evidence."""
    matches = load_match_data(args.evaluation_root, args.output_root, args.gt_margin_sec)
    scoring_runtime = load_scoring_runtime(args.output_root, matches)
    policies = candidate_policies(args.registry_threshold)
    per_policy: dict[str, dict[str, Any]] = {}
    print(f"Evaluating {len(policies)} policies over {len(matches)} matches", flush=True)
    for policy in policies:
        per_match = {match.match_id: evaluate_policy(match, policy) for match in matches}
        per_policy[policy.name] = {
            "policy": asdict(policy),
            "per_match": per_match,
            "aggregate": aggregate_metrics(per_match.values()),
        }

    oracle_policy, oracle_metrics = select_most_selective_full_recall(
        (policy, per_policy[policy.name]["aggregate"]) for policy in policies
    )
    registered_policy = Policy(args.registry_threshold, 0.0, 0.0, 0.0)
    registered_temporal_policy = Policy(args.registry_threshold, 5.0, 3.0, 3.0)

    leave_one_out = []
    heldout_metrics = []
    for holdout in matches:
        calibration_ids = {match.match_id for match in matches if match.match_id != holdout.match_id}
        calibration_candidates = []
        for policy in policies:
            calibration_metrics = aggregate_metrics(
                per_policy[policy.name]["per_match"][match_id]
                for match_id in calibration_ids
            )
            calibration_candidates.append((policy, calibration_metrics))
        selected_policy, calibration_metrics = select_most_selective_full_recall(
            calibration_candidates
        )
        holdout_metrics = per_policy[selected_policy.name]["per_match"][holdout.match_id]
        heldout_metrics.append(holdout_metrics)
        leave_one_out.append(
            {
                "holdout_match": holdout.match_id,
                "selected_policy": asdict(selected_policy),
                "calibration_9match_metrics": calibration_metrics,
                "holdout_metrics": holdout_metrics,
            }
        )

    report = {
        "schema_version": 1,
        "scope": "standalone speculative gameplay-gating feasibility study",
        "pipeline_integration": False,
        "ground_truth_contract": (
            "confirmed synced events grouped by (set,rally), spanning first through last "
            f"event with +/- {args.gt_margin_sec:g}s margin"
        ),
        "match_count": len(matches),
        "policy_count": len(policies),
        "scoring_runtime": scoring_runtime,
        "registered_raw": per_policy[registered_policy.name],
        "registered_temporal": per_policy[registered_temporal_policy.name],
        "oracle_all10": {
            "warning": "selected on all ten evaluation matches; diagnostic, not unbiased",
            "policy": asdict(oracle_policy),
            "aggregate": oracle_metrics,
            "per_match": per_policy[oracle_policy.name]["per_match"],
        },
        "leave_one_match_out": {
            "contract": (
                "for each held-out match, select the least-retaining full-recall policy "
                "using only the other nine matches"
            ),
            "aggregate_holdouts": aggregate_metrics(heldout_metrics),
            "folds": leave_one_out,
        },
    }
    for section_name in ("registered_raw", "registered_temporal"):
        report[section_name]["cost_model"] = gate_cost_model(
            report[section_name]["aggregate"], scoring_runtime
        )
    report["oracle_all10"]["cost_model"] = gate_cost_model(
        report["oracle_all10"]["aggregate"], scoring_runtime
    )
    report["leave_one_match_out"]["cost_model"] = gate_cost_model(
        report["leave_one_match_out"]["aggregate_holdouts"], scoring_runtime
    )
    atomic_write_json(args.output_root / "gameplay_gate_report.json", report)
    write_policy_sweep(args.output_root / "policy_sweep.csv", policies, per_policy)
    write_selected_intervals(
        args.output_root / "oracle_full_recall_intervals.csv",
        matches,
        oracle_policy,
    )
    write_markdown_report(args.output_root / "gameplay_gate_report.md", report)
    print(json.dumps(report["oracle_all10"], indent=2, sort_keys=True), flush=True)
    print(
        json.dumps(report["leave_one_match_out"]["aggregate_holdouts"], indent=2, sort_keys=True),
        flush=True,
    )


def load_scoring_runtime(
    output_root: Path,
    matches: Sequence[MatchData],
) -> dict[str, float | int]:
    """Aggregate measured serial-equivalent gameplay scoring costs."""
    metadata_rows = [
        json.loads(
            (output_root / match.match_id / "score_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        for match in matches
    ]
    video_seconds = sum(float(row["timing"]["duration_sec"]) for row in metadata_rows)
    total_seconds = sum(float(row["timing"]["total_seconds"]) for row in metadata_rows)
    classifier_seconds = sum(
        float(row["timing"]["classifier_seconds"]) for row in metadata_rows
    )
    sampled_frames = sum(int(row["timing"]["sampled_frames"]) for row in metadata_rows)
    return {
        "matches": len(metadata_rows),
        "video_seconds": video_seconds,
        "video_hours": video_seconds / 3600.0,
        "serial_equivalent_first_pass_seconds": total_seconds,
        "serial_equivalent_first_pass_minutes": total_seconds / 60.0,
        "classifier_head_seconds": classifier_seconds,
        "sampled_frames": sampled_frames,
        "source_video_seconds_per_wallclock_second": (
            video_seconds / total_seconds if total_seconds else 0.0
        ),
        "first_pass_wallclock_fraction_of_video_duration": (
            total_seconds / video_seconds if video_seconds else 0.0
        ),
    }


def gate_cost_model(
    metrics: dict[str, float | int],
    scoring_runtime: dict[str, float | int],
) -> dict[str, float | str]:
    """Estimate when first-pass gating repays its measured runtime cost."""
    excluded_fraction = float(metrics["excluded_video_fraction"])
    video_seconds = float(scoring_runtime["video_seconds"])
    gate_seconds = float(scoring_runtime["serial_equivalent_first_pass_seconds"])
    break_even_realtime_factor = (
        excluded_fraction * video_seconds / gate_seconds if gate_seconds else 0.0
    )
    return {
        "assumption": "heavy detector cost is proportional to retained video duration",
        "theoretical_detector_work_avoided_fraction": excluded_fraction,
        "break_even_detector_source_video_seconds_per_wallclock_second": (
            break_even_realtime_factor
        ),
    }


def write_policy_sweep(
    path: Path,
    policies: Sequence[Policy],
    per_policy: dict[str, dict[str, Any]],
) -> None:
    """Write aggregate results for every swept policy."""
    fieldnames = (
        "threshold",
        "merge_gap_sec",
        "pad_before_sec",
        "pad_after_sec",
        "annotated_gameplay_recall",
        "event_recall",
        "rally_full_coverage_recall",
        "retained_video_fraction",
        "retained_annotation_negative_fraction",
        "retained_total_seconds",
        "excluded_video_seconds",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy in policies:
            metrics = per_policy[policy.name]["aggregate"]
            writer.writerow(
                {
                    **asdict(policy),
                    **{name: metrics[name] for name in fieldnames if name in metrics},
                }
            )
    temporary.replace(path)


def write_selected_intervals(path: Path, matches: Sequence[MatchData], policy: Policy) -> None:
    """Write selected diagnostic intervals for timeline review."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("match_id", "segment", "start_sec", "end_sec", "duration_sec"),
        )
        writer.writeheader()
        for match in matches:
            intervals = predicted_intervals(
                match.starts_sec,
                match.scores,
                policy.threshold,
                match.window_sec,
                policy.merge_gap_sec,
                policy.pad_before_sec,
                policy.pad_after_sec,
                match.duration_sec,
            )
            for index, interval in enumerate(intervals, start=1):
                writer.writerow(
                    {
                        "match_id": match.match_id,
                        "segment": index,
                        "start_sec": f"{interval.start_sec:.3f}",
                        "end_sec": f"{interval.end_sec:.3f}",
                        "duration_sec": f"{interval.end_sec - interval.start_sec:.3f}",
                    }
                )
    temporary.replace(path)


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    """Write a compact human-readable summary of the feasibility study."""
    registered = report["registered_raw"]["aggregate"]
    temporal = report["registered_temporal"]["aggregate"]
    oracle = report["oracle_all10"]["aggregate"]
    leave_one_out = report["leave_one_match_out"]["aggregate_holdouts"]
    runtime = report["scoring_runtime"]

    def row(name: str, metrics: dict[str, float | int]) -> str:
        """Format one report table row."""
        return (
            f"| {name} | {100 * float(metrics['annotated_gameplay_recall']):.4f}% | "
            f"{100 * float(metrics['rally_full_coverage_recall']):.2f}% | "
            f"{float(metrics['missed_gameplay_seconds']):.2f} | "
            f"{float(metrics['retained_total_seconds']) / 60:.2f} | "
            f"{float(metrics['retained_annotation_negative_seconds']) / 60:.2f} | "
            f"{100 * float(metrics['excluded_video_fraction']):.2f}% |"
        )

    lines = [
        "# Gameplay Gate Feasibility Study",
        "",
        "This is a standalone speculative evaluation. It does not change or integrate with the pipeline.",
        "",
        f"Ground truth: {report['ground_truth_contract']}.",
        "",
        (
            f"The first pass scored {float(runtime['video_hours']):.2f} hours of video in "
            f"{float(runtime['serial_equivalent_first_pass_minutes']):.2f} serial-equivalent "
            f"minutes ({float(runtime['source_video_seconds_per_wallclock_second']):.2f}x "
            "real time)."
        ),
        "",
        "| Policy | gameplay seconds recalled | fully covered rallies | missed gameplay sec | retained video min | retained annotation-negative min | theoretical detector work excluded |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        row("registry threshold, no temporal postprocessing", registered),
        row("registry threshold, fill 5s gaps + pad 3s", temporal),
        row("all-10 oracle (biased diagnostic)", oracle),
        row("leave-one-match-out transfer", leave_one_out),
        "",
        "The annotation-negative metric treats all time outside margin-expanded synchronized rallies as non-gameplay. The all-10 oracle is not a deployable estimate because it was selected on the same ten matches it reports.",
        "",
        "## Selected all-10 oracle policy",
        "",
        "```json",
        json.dumps(report["oracle_all10"]["policy"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score", help="score raw evaluation videos")
    score.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    score.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    score.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    score.add_argument("--device", default="cuda:0")
    score.add_argument("--sample-fps", type=float, default=1.0)
    score.add_argument("--decode-long-edge", type=int, default=512)
    score.add_argument("--frame-batch-size", type=int, default=32)
    score.add_argument("--classifier-batch-size", type=int, default=2048)
    score.add_argument("--num-shards", type=int, default=1)
    score.add_argument("--shard-index", type=int, default=0)
    score.add_argument("--force", action="store_true")
    score.set_defaults(function=score_command)

    analyze = subparsers.add_parser("analyze", help="evaluate temporal gating policies")
    analyze.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    analyze.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    analyze.add_argument("--gt-margin-sec", type=float, default=3.0)
    analyze.add_argument(
        "--registry-threshold", type=float, default=DEFAULT_REGISTRY_THRESHOLD
    )
    analyze.set_defaults(function=analyze_command)
    return parser


def main() -> None:
    """Run the requested scoring or analysis command."""
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
