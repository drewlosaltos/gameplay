#!/usr/bin/env python3
"""Validate an extracted staged SkyBall gameplay dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


POPULATIONS = {"train", "v2_fixed", "v5_expanded"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-val-metadata", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--video-manifest", type=Path)
    parser.add_argument("--val-video-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Parsed records.
    """
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_video_id(value: object) -> str:
    """Normalize a source filename to its stem.

    Args:
        value: Source video name or ID.

    Returns:
        Normalized source-video ID.
    """
    return Path(str(value).strip()).stem


def population_for(window: dict) -> str:
    """Return the staged population represented by a window.

    Args:
        window: Window metadata record.

    Returns:
        ``train``, ``v2_fixed``, or ``v5_expanded``.
    """
    return str(window.get("validation_cohort") or "train")


def validate_population_membership(
    windows: list[dict],
    expected_fixed: set[str],
    expected_expanded: set[str],
) -> dict[str, set[str]]:
    """Validate split isolation and exact validation source membership.

    Args:
        windows: Extracted window records.
        expected_fixed: Frozen v2 validation source IDs.
        expected_expanded: Added v5 validation source IDs.

    Returns:
        Source-video IDs by population.

    Raises:
        ValueError: If a video leaks across populations or membership differs.
    """
    populations_by_video: dict[str, set[str]] = defaultdict(set)
    for window in windows:
        population = population_for(window)
        if population not in POPULATIONS:
            raise ValueError(f"Unknown staged population: {population}")
        video_id = str(window.get("video_id") or normalize_video_id(window["source_video"]))
        populations_by_video[video_id].add(population)
        expected_split = "train" if population == "train" else "val"
        if window.get("split") != expected_split:
            raise ValueError(f"Window {window.get('window_id')} has inconsistent split/population")
    leaking = sorted(video for video, populations in populations_by_video.items() if len(populations) != 1)
    if leaking:
        raise ValueError("Source videos occur in multiple populations: " + ", ".join(leaking))
    by_population = {
        population: {video for video, memberships in populations_by_video.items() if population in memberships}
        for population in POPULATIONS
    }
    if by_population["v2_fixed"] != expected_fixed:
        raise ValueError("Frozen v2 validation membership does not match the split manifest")
    if by_population["v5_expanded"] != expected_expanded:
        raise ValueError("Expanded v5 validation membership does not match the split manifest")
    return by_population


def semantic_window_signature(window: dict) -> tuple:
    """Build a path-independent semantic signature for a temporal window.

    Args:
        window: Window metadata record.

    Returns:
        Comparable tuple describing sampling and label semantics.
    """
    return (
        window.get("window_id"),
        window.get("video_id") or normalize_video_id(window.get("source_video", "")),
        window.get("class_name"),
        window.get("binary_label"),
        window.get("start_sec"),
        tuple(window.get("times_sec", [])),
        window.get("region_start_sec"),
        window.get("region_end_sec"),
        window.get("start_exclude_sec"),
        window.get("end_exclude_sec"),
        window.get("stride_sec"),
    )


def validate_dataset(dataset_root: Path, base_val_metadata: Path, split_manifest: Path) -> dict:
    """Validate manifests, image parity, split isolation, and frozen validation.

    Args:
        dataset_root: Extracted v5 dataset directory.
        base_val_metadata: v2 metadata directory or windows manifest.
        split_manifest: Versioned v5 split-and-sampling manifest.

    Returns:
        Validation report.

    Raises:
        ValueError: If any dataset invariant fails.
    """
    metadata = dataset_root / "metadata"
    required = [
        metadata / "windows.jsonl",
        metadata / "images.jsonl",
        metadata / "summary.json",
        metadata / "class_to_idx.json",
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise ValueError("Missing required dataset metadata: " + ", ".join(missing_required))

    windows = read_jsonl(metadata / "windows.jsonl")
    images = read_jsonl(metadata / "images.jsonl")
    summary = json.loads((metadata / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    expected_fixed = {normalize_video_id(video) for video in manifest["v2_fixed_videos"]}
    expected_expanded = {
        normalize_video_id(entry.get("source_video", "") if isinstance(entry, dict) else entry)
        for entry in manifest["videos"]
    }
    by_population = validate_population_membership(windows, expected_fixed, expected_expanded)

    if summary.get("window_count") != len(windows):
        raise ValueError("summary.json window_count does not match windows.jsonl")
    if summary.get("image_count") != len(images):
        raise ValueError("summary.json image_count does not match images.jsonl")
    if summary.get("video_count") != len(set().union(*by_population.values())):
        raise ValueError("summary.json video_count does not match source membership")

    expected_policy = manifest["sampling_policy"]
    actual_policy = summary.get("sampling_policy", {})
    for population in POPULATIONS:
        if actual_policy.get(population) != expected_policy.get(population):
            raise ValueError(f"Sampling policy mismatch for {population}")

    frame_paths = []
    for window in windows:
        paths = window.get("frame_paths", [])
        if len(paths) != 4 or len(window.get("times_sec", [])) != 4:
            raise ValueError(f"Window {window.get('window_id')} is not a four-frame sample")
        frame_paths.extend(paths)
    image_paths = [str(image["path"]) for image in images]
    if Counter(frame_paths) != Counter(image_paths):
        raise ValueError("Window frame paths and images.jsonl paths differ")
    missing_images = [path for path in image_paths if not (dataset_root / path).exists()]
    if missing_images:
        raise ValueError(f"Missing {len(missing_images)} referenced images; first: {missing_images[0]}")

    base_windows_path = base_val_metadata / "windows.jsonl" if base_val_metadata.is_dir() else base_val_metadata
    base_windows = [window for window in read_jsonl(base_windows_path) if window.get("split") == "val"]
    frozen_windows = [window for window in windows if population_for(window) == "v2_fixed"]
    base_signatures = sorted(semantic_window_signature(window) for window in base_windows)
    frozen_signatures = sorted(semantic_window_signature(window) for window in frozen_windows)
    if frozen_signatures != base_signatures:
        raise ValueError("Frozen v2 validation windows do not reproduce the base validation semantics")

    population_window_counts = Counter(population_for(window) for window in windows)
    report = {
        "status": "passed",
        "dataset_root": str(dataset_root),
        "window_count": len(windows),
        "image_count": len(images),
        "population_video_counts": {
            population: len(by_population[population]) for population in sorted(POPULATIONS)
        },
        "population_window_counts": dict(sorted(population_window_counts.items())),
        "missing_image_count": 0,
        "frozen_v2_semantic_window_count": len(frozen_windows),
    }
    return report


def manifest_videos(path: Path) -> set[str]:
    """Load normalized video IDs from a generic top-level videos list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = {
        normalize_video_id(entry.get("source_video", "") if isinstance(entry, dict) else entry)
        for entry in payload.get("videos", [])
    }
    videos.discard("")
    if not videos:
        raise ValueError(f"No videos found in {path}")
    return videos


def validate_source_held_out_dataset(
    dataset_root: Path,
    video_manifest: Path,
    val_video_manifest: Path,
) -> dict:
    """Validate a generic source-video-held-out gameplay dataset."""
    metadata = dataset_root / "metadata"
    required = [
        metadata / "windows.jsonl",
        metadata / "images.jsonl",
        metadata / "summary.json",
        metadata / "class_to_idx.json",
    ]
    missing_required = [str(path) for path in required if not path.exists()]
    if missing_required:
        raise ValueError("Missing required dataset metadata: " + ", ".join(missing_required))

    windows = read_jsonl(metadata / "windows.jsonl")
    images = read_jsonl(metadata / "images.jsonl")
    summary = json.loads((metadata / "summary.json").read_text(encoding="utf-8"))
    expected_videos = manifest_videos(video_manifest)
    expected_val = manifest_videos(val_video_manifest)
    if not expected_val < expected_videos:
        raise ValueError("Validation manifest must be a proper subset of the selected population")

    windows_by_video: dict[str, set[str]] = defaultdict(set)
    window_ids: set[str] = set()
    frame_paths: list[str] = []
    for window in windows:
        window_id = str(window["window_id"])
        if window_id in window_ids:
            raise ValueError(f"Duplicate window_id: {window_id}")
        window_ids.add(window_id)
        video_id = str(window.get("video_id") or normalize_video_id(window["source_video"]))
        split = str(window["split"])
        windows_by_video[video_id].add(split)
        expected_split = "val" if video_id in expected_val else "train"
        if split != expected_split:
            raise ValueError(f"Window {window_id} has split {split}, expected {expected_split}")
        times = [float(value) for value in window.get("times_sec", [])]
        paths = list(window.get("frame_paths", []))
        if len(times) != 4 or len(paths) != 4:
            raise ValueError(f"Window {window_id} is not a four-frame sample")
        if times != [round(float(window["start_sec"]) + offset, 3) for offset in range(4)]:
            raise ValueError(f"Window {window_id} has inconsistent one-Hz timestamps")
        if (
            times[0] < float(window["stable_start_sec"])
            or float(window["end_sec"]) > float(window["stable_end_sec"])
        ):
            raise ValueError(f"Window {window_id} falls outside its stable labeled region")
        expected_margin = max(
            float(window["start_exclude_sec"]),
            float(window["end_exclude_sec"]),
        )
        if float(window.get("boundary_margin_sec", -1)) != expected_margin:
            raise ValueError(f"Window {window_id} has inconsistent boundary_margin_sec")
        if int(window.get("sampling_seed", -1)) != int(summary["seed"]):
            raise ValueError(f"Window {window_id} has inconsistent sampling_seed")
        if not Path(window["raw_video_path"]).is_file():
            raise ValueError(f"Window {window_id} references a missing raw video")
        frame_paths.extend(paths)

    leaking = sorted(video for video, splits in windows_by_video.items() if len(splits) != 1)
    if leaking:
        raise ValueError("Source videos occur in multiple splits: " + ", ".join(leaking))
    actual_videos = set(windows_by_video)
    if actual_videos != expected_videos:
        raise ValueError(
            f"Source population mismatch: expected {len(expected_videos)}, got {len(actual_videos)}"
        )
    actual_val = {video for video, splits in windows_by_video.items() if "val" in splits}
    if actual_val != expected_val:
        raise ValueError("Validation source membership does not match the manifest")

    image_paths = [str(image["path"]) for image in images]
    if Counter(frame_paths) != Counter(image_paths):
        raise ValueError("Window frame paths and images.jsonl paths differ")
    if len({str(image["image_id"]) for image in images}) != len(images):
        raise ValueError("Duplicate image_id records found")
    missing_images = [path for path in image_paths if not (dataset_root / path).is_file()]
    if missing_images:
        raise ValueError(f"Missing {len(missing_images)} referenced images; first: {missing_images[0]}")
    if summary.get("window_count") != len(windows) or summary.get("image_count") != len(images):
        raise ValueError("summary.json counts do not match JSONL manifests")
    if summary.get("video_count") != len(actual_videos):
        raise ValueError("summary.json video_count does not match source membership")

    split_counts = Counter(window["split"] for window in windows)
    binary_counts = Counter(
        f"{window['split']}/{'gameplay' if int(window['binary_label']) else 'non_gameplay'}"
        for window in windows
    )
    return {
        "status": "passed",
        "dataset_root": str(dataset_root),
        "window_count": len(windows),
        "image_count": len(images),
        "video_count": len(actual_videos),
        "train_video_count": len(actual_videos - actual_val),
        "val_video_count": len(actual_val),
        "window_counts_by_split": dict(sorted(split_counts.items())),
        "binary_window_counts_by_split": dict(sorted(binary_counts.items())),
        "missing_image_count": 0,
        "source_split_leakage_count": 0,
    }


def main() -> None:
    """Validate a dataset and write its versioned audit report."""
    args = parse_args()
    if args.video_manifest or args.val_video_manifest:
        if not args.video_manifest or not args.val_video_manifest:
            raise SystemExit("--video-manifest and --val-video-manifest must be passed together")
        report = validate_source_held_out_dataset(
            args.dataset_root,
            args.video_manifest,
            args.val_video_manifest,
        )
    else:
        if not args.base_val_metadata or not args.split_manifest:
            raise SystemExit(
                "Pass either the generic video/validation manifests or the staged base/split inputs"
            )
        report = validate_dataset(args.dataset_root, args.base_val_metadata, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
