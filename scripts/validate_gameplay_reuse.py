#!/usr/bin/env python3
"""Validate timestamp-matched hardlink reuse in a gameplay dataset refresh."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--reuse-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def normalize_video_id(value: object) -> str:
    return Path(str(value).strip()).stem


def image_key(row: dict) -> tuple[str, float]:
    video_id = str(row.get("video_id") or normalize_video_id(row.get("source_video", "")))
    return video_id, round(float(row["time_sec"]), 3)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_source_index(reuse_root: Path) -> dict[tuple[str, float], Path]:
    """Index the first existing legacy JPEG for every source/timestamp key."""
    index: dict[tuple[str, float], Path] = {}
    images_path = reuse_root / "metadata" / "images.jsonl"
    with images_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = image_key(row)
            if key in index:
                continue
            path = reuse_root / row["path"]
            if path.exists():
                index[key] = path
    return index


def validate_reuse(dataset_root: Path, reuse_root: Path) -> dict:
    """Prove all timestamp-compatible target JPEGs are hardlinks to legacy JPEGs."""
    dataset_root = dataset_root.resolve()
    reuse_root = reuse_root.resolve()
    if dataset_root == reuse_root:
        raise ValueError("Dataset root and reuse root must differ")

    summary = read_json(dataset_root / "metadata" / "summary.json")
    reuse_summary = summary.get("reuse_summary")
    if not reuse_summary:
        raise ValueError("Dataset summary does not contain reuse evidence")
    if Path(reuse_summary["source_dataset"]).resolve() != reuse_root:
        raise ValueError("Dataset summary reuse source does not match --reuse-root")

    source_summary = read_json(reuse_root / "metadata" / "summary.json")
    for field in ("long_edge", "jpeg_quality"):
        if summary.get(field) != source_summary.get(field):
            raise ValueError(f"Extraction setting differs for {field}")

    source_index = build_source_index(reuse_root)
    expected_reused = 0
    verified_hardlinks = 0
    reused_by_video: Counter[str] = Counter()
    missing_target_count = 0
    non_hardlink_count = 0
    first_failure = None
    images_path = dataset_root / "metadata" / "images.jsonl"
    with images_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = image_key(row)
            source_path = source_index.get(key)
            if source_path is None:
                continue
            expected_reused += 1
            target_path = dataset_root / row["path"]
            if not target_path.exists():
                missing_target_count += 1
                first_failure = first_failure or f"missing target: {target_path}"
                continue
            source_stat = os.stat(source_path)
            target_stat = os.stat(target_path)
            if (source_stat.st_dev, source_stat.st_ino) != (target_stat.st_dev, target_stat.st_ino):
                non_hardlink_count += 1
                first_failure = first_failure or f"not a hardlink: {target_path} -> {source_path}"
                continue
            verified_hardlinks += 1
            reused_by_video[key[0]] += 1

    recorded_reused = int(reuse_summary.get("reused_image_count", -1))
    recorded_hardlinks = int(reuse_summary.get("hardlink_count", -1))
    recorded_copies = int(reuse_summary.get("copied_count", -1))
    if recorded_copies != 0:
        raise ValueError(f"Expected hardlink-only reuse on this filesystem, found {recorded_copies} copies")
    if expected_reused != recorded_reused or verified_hardlinks != recorded_hardlinks:
        raise ValueError(
            "Reuse counts do not match summary: "
            f"expected={expected_reused}, verified={verified_hardlinks}, "
            f"recorded_reused={recorded_reused}, recorded_hardlinks={recorded_hardlinks}"
        )
    if missing_target_count or non_hardlink_count:
        raise ValueError(
            f"Reuse validation failed: missing={missing_target_count}, "
            f"non_hardlinks={non_hardlink_count}; first={first_failure}"
        )

    return {
        "status": "passed",
        "dataset_root": str(dataset_root),
        "reuse_root": str(reuse_root),
        "source_timestamp_count": len(source_index),
        "expected_reused_image_count": expected_reused,
        "verified_hardlink_count": verified_hardlinks,
        "reused_video_count": len(reused_by_video),
        "missing_target_count": 0,
        "non_hardlink_count": 0,
        "copied_fallback_count": 0,
    }


def main() -> None:
    args = parse_args()
    report = validate_reuse(args.dataset_root, args.reuse_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
