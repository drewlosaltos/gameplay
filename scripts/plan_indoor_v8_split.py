#!/usr/bin/env python3
"""Freeze val20 and select 30 previously unseen v8 sources for val50."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_V8 = Path(
    "/mnt/t/data/vball/skyball/gameplay/ground_truth/indoor/"
    "skyball-gameplay-finetune/v8"
)
DEFAULT_BASE = Path(
    "/mnt/t/data/vball/skyball/gameplay/human_app_v5_val20_20260925"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_tertiles(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda video: (values[video], video))
    return {video: min(2, 3 * position // len(ordered)) for position, video in enumerate(ordered)}


def select_group(
    videos: list[str],
    count: int,
    duration: dict[str, float],
    gameplay_fraction: dict[str, float],
    program: dict[str, str],
    rng: random.Random,
) -> tuple[list[str], dict[str, int]]:
    duration_bucket = rank_tertiles({video: duration[video] for video in videos})
    gameplay_bucket = rank_tertiles({video: gameplay_fraction[video] for video in videos})
    groups: dict[str, list[str]] = defaultdict(list)
    for video in videos:
        groups[f"duration{duration_bucket[video]}_gameplay{gameplay_bucket[video]}"].append(video)
    exact = {key: count * len(group) / len(videos) for key, group in groups.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remaining = count - sum(quotas.values())
    for key in sorted(groups, key=lambda item: (-(exact[item] - quotas[item]), item))[:remaining]:
        quotas[key] += 1
    selected: list[str] = []
    used_programs: set[str] = set()
    for key in sorted(groups):
        candidates = sorted(groups[key])
        rng.shuffle(candidates)
        for _ in range(quotas[key]):
            position = next(
                (index for index, video in enumerate(candidates)
                 if not program[video] or program[video] not in used_programs),
                0,
            )
            video = candidates.pop(position)
            selected.append(video)
            if program[video]:
                used_programs.add(program[video])
    assert len(selected) == count and len(set(selected)) == count
    return selected, quotas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8", type=Path, default=DEFAULT_V8)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()

    release = json.loads((args.v8 / "manifest.json").read_text())
    selected_ids = set(release["training_video_ids"])
    assert len(selected_ids) == 1081 and release["version"] == "v8"
    base = json.loads((args.base / "metadata/summary.json").read_text())
    val20 = set(base["val_videos"])
    old_train = set(base["train_videos"])
    assert len(val20) == 20 and val20 <= selected_ids
    new = selected_ids - val20 - old_train
    assert len(new) == 499

    duration: dict[str, float] = defaultdict(float)
    gameplay: dict[str, float] = defaultdict(float)
    program: dict[str, str] = {}
    rows = 0
    with (args.v8 / "phases.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            video = row["video_id"]
            seconds = float(row["duration_sec"])
            duration[video] += seconds
            if row["category"] == "gameplay":
                gameplay[video] += seconds
            program[video] = row["program_id"]
            rows += 1
    assert rows == 17284 and set(duration) == selected_ids
    fraction = {video: gameplay[video] / duration[video] for video in new}
    rng = random.Random(args.seed)
    vid, vid_quotas = select_group(
        sorted(video for video in new if video.startswith("vid_")),
        22, duration, fraction, program, rng,
    )
    original, original_quotas = select_group(
        sorted(video for video in new if video.startswith("original_")),
        8, duration, fraction, program, rng,
    )
    new30 = set(vid + original)
    assert len(new30) == 30 and not new30 & (val20 | old_train)
    train = selected_ids - val20 - new30
    assert len(train) == 1031

    payload = {
        "version": "v8_val50_v1",
        "selection_seed": args.seed,
        "selection_policy": (
            "Keep the exact v5 val20; select 22 vid and 8 original sources unseen "
            "in v5, stratified by within-type duration and gameplay-share tertiles, "
            "favoring distinct known program IDs."
        ),
        "source_label_status": (
            "The 30 new sources have model-reviewed labels pending human acceptance; "
            "report their scores provisionally until the annotations are reviewed."
        ),
        "v8_manifest_sha256": sha256(args.v8 / "manifest.json"),
        "v8_phases_sha256": sha256(args.v8 / "phases.csv"),
        "base_windows_sha256": sha256(args.base / "metadata/windows.jsonl"),
        "val20": sorted(val20),
        "new30": sorted(new30),
        "val50": sorted(val20 | new30),
        "train": sorted(train),
        "stratum_quotas": {"vid": vid_quotas, "original": original_quotas},
        "counts": {"v8_sources": len(selected_ids), "train": len(train),
                   "val20": len(val20), "new30": len(new30), "val50": 50},
    }
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace frozen split: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **payload["counts"]}))


if __name__ == "__main__":
    main()
