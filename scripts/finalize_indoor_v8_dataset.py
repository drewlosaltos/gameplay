#!/usr/bin/env python3
"""Validate v8 extraction and publish its train/val JSONL manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from itertools import islice
from pathlib import Path


DEFAULT_WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1")
REMOTE_LISTING = Path(
    "/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/shared_video_listing.json"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    plan = json.loads((args.work / "plan.json").read_text())
    overrides_path = args.work / "raw_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    split = json.loads(Path(plan["split_manifest"]).read_text())
    early_reprobe_path = Path(plan["reused_early_raw_reprobe"]) if "reused_early_raw_reprobe" in plan else None
    if early_reprobe_path is not None:
        early_reprobe = json.loads(early_reprobe_path.read_text())
        if early_reprobe.get("status") != "pass" or early_reprobe.get("reused_only_early_sources") != 46:
            raise RuntimeError("The 46 reused-only early raw reads have not passed re-probe")
    suspect_audit_path = Path(plan["suspect_image_audit"]) if "suspect_image_audit" in plan else None
    if suspect_audit_path is not None:
        suspect_audit = json.loads(suspect_audit_path.read_text())
        if (suspect_audit.get("status") != "complete" or suspect_audit.get("suspect_sources") != 11
                or suspect_audit.get("fully_decoded_jpegs", 0) <= 0
                or suspect_audit.get("dataset") != plan["dataset"]):
            raise RuntimeError("Re-extracted JPEG audit is incomplete or for a different dataset")
    dataset = Path(plan["dataset"])
    base = Path(plan["base_dataset"])
    target = dataset / "metadata"
    staging = args.work / "metadata_build_v1"
    if target.exists() or staging.exists():
        raise FileExistsError("Metadata target or staging already exists")
    if not (args.work / "base_reuse_complete.json").exists():
        raise RuntimeError("Base image reuse has not completed")
    checks = {}
    for path in (args.work / "video_checks").glob("*.json"):
        row = json.loads(path.read_text())
        if row["status"] != "complete":
            raise RuntimeError(f"Incomplete video check: {path}")
        checks[row["video_id"]] = row
    expected_checks = set(split["train"]) | set(split["new30"])
    if set(checks) != expected_checks:
        raise RuntimeError(
            f"Video checks differ: missing {len(expected_checks-set(checks))}, "
            f"extra {len(set(checks)-expected_checks)}"
        )
    remote = {row["Name"]: (int(row["Size"]),
              datetime.fromisoformat(row["ModTime"].replace("Z", "+00:00")).timestamp())
              for row in json.loads(REMOTE_LISTING.read_text())}
    for video_id in set(split["train"]) | set(split["val50"]):
        path = Path(overrides.get(video_id,
                                  str(Path(plan["raw_root"]) / f"{video_id}.mp4")))
        if (not path.is_file() or path.stat().st_size != remote[path.name][0]
                or abs(path.stat().st_mtime - remote[path.name][1]) > 1.0):
            raise RuntimeError(f"Raw-video size/mtime is wrong: {path}")
    dropped = {name for row in checks.values() for name in row["dropped_audio_tail_windows"]}
    if sum(row["planned_windows"] for row in checks.values()) != plan["planned_new_windows"]:
        raise RuntimeError("Planned windows do not match per-video checks")

    staging.mkdir()
    window_path = staging / "windows.jsonl"
    image_path = staging / "images.jsonl"
    counts: Counter[str] = Counter()
    cohorts: Counter[str] = Counter()
    sources: dict[str, set[str]] = {"train": set(), "val20": set(), "new30": set()}
    seen_windows: set[str] = set()
    image_count = 0

    def append_window(row: dict, windows, images) -> None:
        nonlocal image_count
        window_id = row["window_id"]
        if window_id in seen_windows:
            raise RuntimeError(f"Duplicate window ID: {window_id}")
        seen_windows.add(window_id)
        split_name = row["split"]
        video_id = row["video_id"]
        if split_name == "train":
            if video_id not in split["train"]:
                raise RuntimeError(f"Train source outside train split: {video_id}")
            cohort = "train"
        elif video_id in split["val20"]:
            cohort = "val20"
        elif video_id in split["new30"]:
            cohort = "new30"
        else:
            raise RuntimeError(f"Unexpected validation source: {video_id}")
        if len(row["frame_paths"]) != 4 or len(row["times_sec"]) != 4:
            raise RuntimeError(f"Not a four-frame window: {window_id}")
        if [round(float(value), 3) for value in row["times_sec"]] != [
            round(float(row["start_sec"]) + offset, 3) for offset in range(4)
        ]:
            raise RuntimeError(f"Non-1-Hz sample: {window_id}")
        if min(row["times_sec"]) < row["region_start_sec"] - .01 or max(row["times_sec"]) > row["region_end_sec"] + .01:
            raise RuntimeError(f"Window outside labeled region: {window_id}")
        if (
            float(row["start_sec"]) < float(row["stable_start_sec"]) - .01
            or float(row["end_sec"]) > float(row["stable_end_sec"]) + .01
        ):
            raise RuntimeError(f"Window crosses boundary margin: {window_id}")
        if int(row["binary_label"]) != int(row["class_name"] == "gameplay"):
            raise RuntimeError(f"Binary label disagrees with class: {window_id}")
        for index, (relative, stamp) in enumerate(zip(row["frame_paths"], row["times_sec"], strict=True)):
            image = {
                "image_id": f"{window_id}_s{index}",
                "window_id": window_id,
                "path": relative,
                "frame_index": index,
                "time_sec": stamp,
                "source_video": row["source_video"],
                "raw_video_path": row["raw_video_path"],
                "class_name": row["class_name"],
                "binary_label": row["binary_label"],
                "split": split_name,
                "validation_cohort": row.get("validation_cohort"),
            }
            images.write(json.dumps(image, sort_keys=True) + "\n")
            image_count += 1
        windows.write(json.dumps(row, sort_keys=True) + "\n")
        counts[f"{split_name}/{row['class_name']}"] += 1
        cohorts[cohort] += 1
        sources[cohort].add(video_id)

    def valid_image(path: Path) -> bool:
        try:
            metadata = os.stat(path)
        except OSError:
            return False
        return stat.S_ISREG(metadata.st_mode) and metadata.st_size > 0

    def verify_and_append(rows, windows, images, pool: ThreadPoolExecutor) -> None:
        iterator = iter(rows)
        while batch := list(islice(iterator, 128)):
            paths = [dataset / relative for row in batch for relative in row["frame_paths"]]
            for path, valid in zip(paths, pool.map(valid_image, paths), strict=True):
                if not valid:
                    raise RuntimeError(f"Missing/empty JPEG: {path}")
            for row in batch:
                append_window(row, windows, images)

    with window_path.open("w") as windows, image_path.open("w") as images, ThreadPoolExecutor(max_workers=16) as pool:
        with (base / "metadata/windows.jsonl").open() as handle:
            verify_and_append(
                (row for line in handle if (row := json.loads(line))["split"] == "val"),
                windows, images, pool,
            )
        with (args.work / "planned_windows.jsonl").open() as handle:
            verify_and_append(
                (row for line in handle if (row := json.loads(line))["window_id"] not in dropped),
                windows, images, pool,
            )

    if cohorts["val20"] != plan["frozen_val20_windows"]:
        raise RuntimeError("Frozen val20 window count changed")
    if sum(cohorts.values()) + len(dropped) != plan["planned_new_windows"] + plan["frozen_val20_windows"]:
        raise RuntimeError("Final window total does not match plan and dropped tails")
    for cohort, expected in (("train",split["train"]),("val20",split["val20"]),("new30",split["new30"])):
        if sources[cohort] != set(expected):
            raise RuntimeError(f"Source membership differs for {cohort}")
    if sources["train"] & (sources["val20"] | sources["new30"]):
        raise RuntimeError("Train/validation source leakage")
    (staging / "class_to_idx.json").write_bytes((base / "metadata/class_to_idx.json").read_bytes())
    summary = {
        "version": plan["version"],
        "dataset_root": str(dataset),
        "source_release": plan["release"],
        "split_manifest": plan["split_manifest"],
        "source_video_counts": {key: len(value) for key, value in sources.items()},
        "window_counts_by_split_class": dict(counts),
        "window_counts_by_cohort": dict(cohorts),
        "window_count": len(seen_windows),
        "image_count": image_count,
        "video_count": sum(len(value) for value in sources.values()),
        "train_video_count": len(sources["train"]),
        "val_video_count": len(sources["val20"]) + len(sources["new30"]),
        "source_videos_by_cohort": {key: sorted(value) for key, value in sources.items()},
        "dropped_audio_tail_windows": sorted(dropped),
        "frozen_val20": {
            "source_dataset": str(base),
            "windows_sha256": plan["base_windows_sha256"],
        },
        "input_sha256": plan["input_sha256"],
        "split_sha256": plan["split_sha256"],
        "windows_sha256": digest(window_path),
        "images_sha256": digest(image_path),
        "label_status": split["source_label_status"],
        "review_clips": "/mnt/t/output/skyball_gameplay/v8_workflow_20261006/review_clips_v8_binary_balanced_r2/summary.json",
        "reused_early_raw_reprobe": str(early_reprobe_path) if early_reprobe_path else None,
        "suspect_image_audit": str(suspect_audit_path) if suspect_audit_path else None,
    }
    (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    os.rename(staging, target)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
