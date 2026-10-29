#!/usr/bin/env python3
"""Plan a source-held-out indoor v8 gameplay dataset from the Drive CSV release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

import build_gameplay_v0_dataset as builder


DEFAULT_RELEASE = Path(
    "/mnt/t/data/vball/skyball/gameplay/ground_truth/indoor/"
    "skyball-gameplay-finetune/v8"
)
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "validation/gameplay_indoor_v8_val50_v1.json"
DEFAULT_BASE = Path("/mnt/t/data/vball/skyball/gameplay/human_app_v5_val20_20260925")
DEFAULT_DATASET = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006")
DEFAULT_WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1")
DEFAULT_RAW = Path("/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def effective_class(category: str, activity: str) -> str:
    """Use reviewed activity for binary play; fall back to phase on human rows."""
    category = category.strip().lower()
    activity = activity.strip().lower()
    if category == "unclassified":
        return "unclassified"
    if not activity:
        return builder.class_dir_for(category)
    if activity.startswith("active rally") or activity.startswith("active play"):
        return "gameplay"
    if category == "gameplay":
        return "timeout" if "between-rally delay" in activity else "prematch"
    return builder.class_dir_for(category)


def validate_rows(rows: list[dict[str, str]], selected: set[str]) -> list[list[object]]:
    by_video: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        video = row["video_id"]
        assert video in selected
        start = float(row["start_sec"])
        end = float(row["end_sec"])
        duration = float(row["duration_sec"])
        assert start >= 0 and end > start and abs(end - start - duration) < .1, row
        by_video[video].append((start, end))
    assert set(by_video) == selected
    gaps = []
    for video, regions in by_video.items():
        regions.sort()
        for (_, left_end), (right_start, _) in zip(regions, regions[1:]):
            assert right_start >= left_end - .02, (video, left_end, right_start)
            if right_start > left_end + .02:
                gaps.append([video, left_end, right_start])
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    args = parser.parse_args()
    if args.work.exists() or args.dataset.exists():
        raise FileExistsError("Use new versioned work and dataset directories")
    release = json.loads((args.release / "manifest.json").read_text())
    split = json.loads(args.split.read_text())
    selected = set(release["training_video_ids"])
    train = set(split["train"])
    val20 = set(split["val20"])
    new30 = set(split["new30"])
    assert selected == train | val20 | new30
    assert not (train & val20 or train & new30 or val20 & new30)
    assert digest(args.release / "manifest.json") == split["v8_manifest_sha256"]
    assert digest(args.release / "phases.csv") == split["v8_phases_sha256"]
    assert digest(args.base / "metadata/windows.jsonl") == split["base_windows_sha256"]

    with (args.release / "phases.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == release["cumulative_segments"] == 17284
    gaps = validate_rows(rows, selected)
    by_row = {index: row for index, row in enumerate(rows)}
    label_rows = []
    class_changes: Counter[str] = Counter()
    for index, row in enumerate(rows):
        video = row["video_id"]
        if video in val20:
            continue
        class_dir = effective_class(row["category"], row["activity_category"])
        coarse_class = "unclassified" if row["category"] == "unclassified" else builder.class_dir_for(row["category"])
        if class_dir != coarse_class:
            class_changes[f"{coarse_class}->{class_dir}"] += 1
        label_rows.append({
            "row_index": index,
            "source_video": video + ".mp4",
            "video_id": video,
            "category": row["category"],
            "category_norm": "gameplay" if class_dir == "gameplay" else (
                "pre match" if class_dir == "prematch" else class_dir
            ),
            "class_dir": class_dir,
            "start_sec": float(row["start_sec"]),
            "end_sec": float(row["end_sec"]),
        })
    labels = pd.DataFrame(label_rows)
    regions = builder.build_regions(labels, 20, 5)
    train_regions = [region for region in regions if region.video_id in train]
    expanded_regions = [region for region in regions if region.video_id in new30]
    populations = [
        (builder.build_windows(train_regions, 4, 20, 10), "train", None),
        (builder.build_windows(expanded_regions, 4, 10, 10), "val", "v8_new30"),
    ]

    args.work.mkdir(parents=True, exist_ok=False)
    groups: dict[str, list[dict]] = defaultdict(list)
    counts: Counter[str] = Counter()
    with (args.work / "planned_windows.jsonl").open("x") as output:
        for windows, population, cohort in populations:
            for window in windows:
                if window["class_name"] == "unclassified":
                    continue
                video = window["video_id"]
                row = by_row[window["source_row_index"]]
                window.update(
                    split=population,
                    validation_cohort=cohort,
                    boundary_margin_sec=max(window["start_exclude_sec"], window["end_exclude_sec"]),
                    sampling_seed=20261006,
                    raw_video_path=str(args.raw / (video + ".mp4")),
                    annotation_activity_category=row["activity_category"],
                    annotation_labeled_by=row["labeled_by"],
                    annotation_snapshot="indoor_v8_20261006",
                    annotation_provenance=(
                        "model_reviewed_pending_human_acceptance"
                        if row["labeled_by"] == "codex-model-review" else "inherited_human"
                    ),
                    decoded_grid_times_sec=[round(time * 2) / 2 for time in window["times_sec"]],
                    frame_paths=[
                        str(Path(population) / window["class_name"] / f"{window['window_id']}_s{i}.jpg")
                        for i in range(4)
                    ],
                )
                output.write(json.dumps(window, sort_keys=True) + "\n")
                groups[video].append(window)
                counts[population + "/" + window["class_name"]] += 1
    assert set(groups) == train | new30, (len(groups), len(train | new30))
    per_video = args.work / "windows_by_video"
    per_video.mkdir()
    for video, windows in groups.items():
        with (per_video / f"{video}.jsonl").open("x") as output:
            for window in windows:
                output.write(json.dumps(window, sort_keys=True) + "\n")
    base_summary = json.loads((args.base / "metadata/summary.json").read_text())
    plan = {
        "version": "v8_val50_v1",
        "release": str(args.release),
        "split_manifest": str(args.split),
        "base_dataset": str(args.base),
        "dataset": str(args.dataset),
        "work": str(args.work),
        "raw_root": str(args.raw),
        "selected_sources": len(selected),
        "train_sources": len(train),
        "frozen_val20_sources": len(val20),
        "new_val30_sources": len(new30),
        "frozen_val20_windows": base_summary["frozen_validation_windows"],
        "planned_new_windows": sum(counts.values()),
        "planned_new_images": 4 * sum(counts.values()),
        "new_window_counts": dict(counts),
        "activity_override_counts": dict(class_changes),
        "unclassified_rows": sum(row["category"] == "unclassified" for row in rows),
        "label_gaps": gaps,
        "input_sha256": {name: digest(args.release / name) for name in ("manifest.json", "phases.csv", "videos.txt")},
        "split_sha256": digest(args.split),
        "base_windows_sha256": split["base_windows_sha256"],
    }
    (args.work / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({key: value for key, value in plan.items() if key != "label_gaps"}, indent=2))


if __name__ == "__main__":
    main()
