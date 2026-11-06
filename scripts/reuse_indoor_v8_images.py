#!/usr/bin/env python3
"""Hardlink exact v5 decoded frames into the versioned v8 dataset."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path


DEFAULT_WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1")


def link_or_verify(source: Path, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
        return True
    except FileExistsError:
        if not os.path.samefile(source, dest):
            raise ValueError(f"Existing v8 image is not the expected hardlink: {dest}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    plan = json.loads((args.work / "plan.json").read_text())
    base = Path(plan["base_dataset"])
    dataset = Path(plan["dataset"])
    if (args.work / "base_reuse_complete.json").exists():
        raise FileExistsError("Base reuse already recorded")
    assert not (dataset / "metadata/windows.jsonl").exists()

    start = time.monotonic()
    by_video_time: dict[tuple[str, int], Path] = {}
    frozen = 0
    existing = 0
    with (base / "metadata/images.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            source = base / row["path"]
            if row["split"] == "val":
                if link_or_verify(source, dataset / row["path"]):
                    frozen += 1
                else:
                    existing += 1
            else:
                key = (row["video_id"], round(float(row.get("decoded_grid_time_sec", row["time_sec"])) * 2))
                by_video_time.setdefault(key, source)
    assert frozen + existing == 19312

    matched = 0
    already = 0
    unmatched = 0
    by_split: Counter[str] = Counter()
    with (args.work / "planned_windows.jsonl").open() as handle:
        for index, line in enumerate(handle, start=1):
            row = json.loads(line)
            video = row["video_id"]
            for stamp, relative in zip(row["decoded_grid_times_sec"], row["frame_paths"], strict=True):
                source = by_video_time.get((video, round(float(stamp) * 2)))
                if source is None:
                    unmatched += 1
                    continue
                if link_or_verify(source, dataset / relative):
                    matched += 1
                else:
                    already += 1
                by_split[row["split"]] += 1
            if index % 10000 == 0:
                print(json.dumps({"windows": index, "matched_images": matched,
                                  "unmatched_images": unmatched}), flush=True)
    report = {
        "status": "complete",
        "base_dataset": str(base),
        "dataset": str(dataset),
        "frozen_val20_images_linked": frozen,
        "frozen_val20_existing_verified": existing,
        "new_window_images_linked": matched,
        "new_window_images_existing_verified": already,
        "new_window_images_unmatched": unmatched,
        "reused_by_split": dict(by_split),
        "elapsed_sec": round(time.monotonic() - start, 1),
    }
    assert matched + already + unmatched == plan["planned_new_images"]
    (args.work / "base_reuse_complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
