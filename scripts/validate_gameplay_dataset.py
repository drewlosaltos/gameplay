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
    parser.add_argument("--base-val-metadata", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
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


def main() -> None:
    """Validate a dataset and write its versioned audit report."""
    args = parse_args()
    report = validate_dataset(args.dataset_root, args.base_val_metadata, args.split_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
