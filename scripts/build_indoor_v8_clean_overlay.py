#!/usr/bin/env python3
"""Build a manifest-only v8 view that drops confirmed train/val re-encodes.

Reads a finalized v8 dataset's metadata and the leak audit, removes the audit's
training videos from train and its validation videos from val, and writes
metadata/ plus README.txt to a new versioned root. Image folders are not
written: hardlink the base train/val folders into the new root, and train from
metadata/windows.jsonl only, since those folders still hold the excluded frames.

Built v8_val48_trainclean_overlay_20261006_r4 (prod gameplay training data) from
the r2 metadata now kept in its provenance/v8_val50_metadata folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT = REPO / "validation/gameplay_indoor_v8_val50_v1.json"
DEFAULT_LEAK_AUDIT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_validation_leak_audit.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-metadata", type=Path, required=True,
                        help="metadata/ folder of the finalized base dataset")
    parser.add_argument("--base-dataset", required=True,
                        help="base dataset root, recorded in the summary")
    parser.add_argument("--output", type=Path, required=True, help="new versioned dataset root")
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--split-name", default=None,
                        help="split path recorded in split.json (default: --split)")
    parser.add_argument("--leak-audit", type=Path, default=DEFAULT_LEAK_AUDIT)
    parser.add_argument("--version", default="v8_val48_trainclean_overlay_v1")
    args = parser.parse_args()

    leak = json.loads(args.leak_audit.read_text())
    if leak["status"] != "confirmed_validation_leak":
        raise RuntimeError("Leak audit has no confirmed re-encodes")
    exclude_train = {pair["training_video"] for pair in leak["confirmed_reencoded_pairs"]}
    exclude_val = {pair["validation_video"] for pair in leak["confirmed_reencoded_pairs"]}
    if args.output.exists():
        raise FileExistsError(f"Use a new versioned dataset root: {args.output}")
    base_summary = json.loads((args.base_metadata / "summary.json").read_text())
    split = json.loads(args.split.read_text())
    if not exclude_train <= set(split["train"]):
        raise RuntimeError("An excluded source is not in the training split")
    if not exclude_val <= set(split["val20"]) or exclude_train & set(split["val50"]):
        raise RuntimeError("Confirmed re-encodes disagree with frozen validation membership")
    split["version"] = args.version
    split["base_split_manifest"] = args.split_name or str(args.split)
    split["excluded_train_reencodes"] = sorted(exclude_train)
    split["excluded_validation_reencodes"] = sorted(exclude_val)
    split["validation_leak_audit"] = str(args.leak_audit)
    split["train"] = [name for name in split["train"] if name not in exclude_train]
    split["val20"] = [name for name in split["val20"] if name not in exclude_val]
    split["val50"] = [name for name in split["val50"] if name not in exclude_val]
    split["counts"]["train"] = len(split["train"])
    split["counts"]["val20"] = len(split["val20"])
    split["counts"]["val50"] = len(split["val50"])
    split["counts"]["sources_in_view"] = len(split["train"]) + len(split["val50"])

    building = Path(str(args.output) + ".building")
    metadata = building / "metadata"
    metadata.mkdir(parents=True)
    split_path = metadata / "split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    shutil.copy2(args.base_metadata / "class_to_idx.json", metadata / "class_to_idx.json")

    def excluded(split_name: str, vid: str) -> bool:
        return (split_name == "train" and vid in exclude_train) or (split_name == "val" and vid in exclude_val)

    class_counts = Counter()
    cohort_counts = Counter()
    seen = {"train": set(), "val20": set(), "new30": set()}
    removed_windows = Counter()
    val50 = set(split["val50"])
    with (args.base_metadata / "windows.jsonl").open() as source, (metadata / "windows.jsonl").open("w") as target:
        for line in source:
            row = json.loads(line)
            vid = row["video_id"]
            if excluded(row["split"], vid):
                removed_windows[vid] += 1
                continue
            target.write(line)
            class_counts[f"{row['split']}/{row['class_name']}"] += 1
            cohort = "train" if row["split"] == "train" else "val20" if vid in split["val20"] else "new30"
            if row["split"] == "val" and vid not in val50:
                raise RuntimeError(f"Validation source outside split: {vid}")
            cohort_counts[cohort] += 1
            seen[cohort].add(vid)
    removed_images = Counter()
    image_count = 0
    with (args.base_metadata / "images.jsonl").open() as source, (metadata / "images.jsonl").open("w") as target:
        for line in source:
            row = json.loads(line)
            vid = Path(row["source_video"]).stem
            if excluded(row["split"], vid):
                removed_images[vid] += 1
                continue
            target.write(line)
            image_count += 1
    if (seen["train"] != set(split["train"]) or seen["val20"] != set(split["val20"])
            or seen["new30"] != set(split["new30"])):
        raise RuntimeError("Manifest source membership differs from clean split")
    if set(removed_windows) != exclude_train | exclude_val:
        raise RuntimeError(f"An excluded source had no windows: {removed_windows}")
    if any(removed_images[vid] != 4 * count for vid, count in removed_windows.items()):
        raise RuntimeError("Removed image counts differ from four-frame windows")
    if image_count != 4 * sum(cohort_counts.values()):
        raise RuntimeError("Image and window counts disagree")

    summary = dict(base_summary)
    summary.update({
        "version": args.version, "dataset_root": str(args.output),
        "split_manifest": str(args.output / "metadata/split.json"),
        "split_sha256": sha256(split_path),
        "source_video_counts": {name: len(values) for name, values in seen.items()},
        "source_videos_by_cohort": {name: sorted(values) for name, values in seen.items()},
        "window_counts_by_split_class": dict(class_counts),
        "window_counts_by_cohort": dict(cohort_counts),
        "window_count": sum(cohort_counts.values()), "image_count": image_count,
        "video_count": sum(len(values) for values in seen.values()),
        "train_video_count": len(seen["train"]), "val_video_count": len(seen["val20"]) + len(seen["new30"]),
        "windows_sha256": sha256(metadata / "windows.jsonl"),
        "images_sha256": sha256(metadata / "images.jsonl"),
        "training_view": "manifest-only temporal view; train/val image folders hold the base dataset's frames",
        "online_selection": "clean val48 F1; full frozen val20 and val50 scored separately from the base after training",
        "excluded_train_reencodes": sorted(exclude_train),
        "excluded_validation_reencodes": sorted(exclude_val),
        "removed_windows": dict(sorted(removed_windows.items())),
        "frozen_val20": {"source_dataset": args.base_dataset, "source_count": len(seen["val20"]),
                         "window_count": cohort_counts["val20"],
                         "note": f"Clean val{len(seen['val20'])} subset; full frozen val20 remains in the base"},
        "validation_leak_audit": str(args.leak_audit),
    })
    (metadata / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (building / "README.txt").write_text(
        "Manifest-only four-frame training view. Hardlink the base dataset's train/val folders "
        f"here ({args.base_dataset}); they include frames of the excluded re-encoded videos. "
        "Only metadata/windows.jsonl and metadata/images.jsonl define the clean population. "
        "Do not use folder scans or ImageFolder. See metadata/summary.json and the leak audit.\n"
    )
    building.rename(args.output)
    print(json.dumps({"dataset": str(args.output), "window_count": summary["window_count"],
                      "image_count": image_count, "removed_windows": summary["removed_windows"],
                      "windows_sha256": summary["windows_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
