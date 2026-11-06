#!/usr/bin/env python3
"""Build a manifest-only v8 overlay with clean val48 checkpoint selection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

BASE = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")
FINAL = Path("/mnt/t/data/vball/skyball/gameplay/v8_val48_trainclean_overlay_20261006_r4")
BUILDING = Path(str(FINAL) + ".building")
WORK = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
SPLIT = Path("/home/atao/vsdevel/skyball/gameplay/validation/gameplay_indoor_v8_val50_v1.json")
EXCLUDE_TRAIN = {"vid_mtxwk77pba622d", "vid_mu31qed1aaf075"}
EXCLUDE_VAL = {"vid_mtyozrerb65868", "vid_mtzaxwm0617fe8"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    leak = json.loads((WORK / "v8_validation_leak_audit.json").read_text())
    if leak["status"] != "confirmed_validation_leak" or {
        pair["training_video"] for pair in leak["confirmed_reencoded_pairs"]
    } != EXCLUDE_TRAIN or {
        pair["validation_video"] for pair in leak["confirmed_reencoded_pairs"]
    } != EXCLUDE_VAL:
        raise RuntimeError("Exclusions disagree with confirmed re-encode audit")
    if FINAL.exists() or BUILDING.exists():
        raise FileExistsError("Versioned training overlay already exists")
    base_summary = json.loads((BASE / "metadata/summary.json").read_text())
    split = json.loads(SPLIT.read_text())
    if not EXCLUDE_TRAIN <= set(split["train"]):
        raise RuntimeError("An excluded source is not in the training split")
    if not EXCLUDE_VAL <= set(split["val20"]) or EXCLUDE_TRAIN & set(split["val50"]):
        raise RuntimeError("Confirmed re-encodes disagree with frozen validation membership")
    split["version"] = "v8_val48_trainclean_overlay_v1"
    split["base_split_manifest"] = str(SPLIT)
    split["excluded_train_reencodes"] = sorted(EXCLUDE_TRAIN)
    split["excluded_validation_reencodes"] = sorted(EXCLUDE_VAL)
    split["validation_leak_audit"] = str(WORK / "v8_validation_leak_audit.json")
    split["train"] = [name for name in split["train"] if name not in EXCLUDE_TRAIN]
    split["val20"] = [name for name in split["val20"] if name not in EXCLUDE_VAL]
    split["val50"] = [name for name in split["val50"] if name not in EXCLUDE_VAL]
    split["counts"]["train"] = len(split["train"])
    split["counts"]["val20"] = len(split["val20"])
    split["counts"]["val50"] = len(split["val50"])
    split["counts"]["sources_in_view"] = len(split["train"]) + len(split["val50"])
    BUILDING.mkdir()
    metadata = BUILDING / "metadata"
    metadata.mkdir()
    for name in ("train", "val"):
        os.symlink(BASE / name, BUILDING / name, target_is_directory=True)
    split_path = metadata / "split.json"
    split_path.write_text(json.dumps(split, indent=2) + "\n")
    shutil.copy2(BASE / "metadata/class_to_idx.json", metadata / "class_to_idx.json")
    class_counts = Counter()
    cohort_counts = Counter()
    seen = {"train": set(), "val20": set(), "new30": set()}
    removed_windows = Counter()
    with (BASE / "metadata/windows.jsonl").open() as source, (metadata / "windows.jsonl").open("w") as target:
        for line in source:
            row = json.loads(line)
            vid = row["video_id"]
            if ((row["split"] == "train" and vid in EXCLUDE_TRAIN)
                    or (row["split"] == "val" and vid in EXCLUDE_VAL)):
                removed_windows[vid] += 1
                continue
            target.write(line)
            class_counts[f"{row['split']}/{row['class_name']}"] += 1
            cohort = "train" if row["split"] == "train" else "val20" if vid in split["val20"] else "new30"
            if row["split"] == "val" and vid not in set(split["val50"]):
                raise RuntimeError(f"Validation source outside split: {vid}")
            cohort_counts[cohort] += 1
            seen[cohort].add(vid)
    removed_images = Counter()
    image_count = 0
    with (BASE / "metadata/images.jsonl").open() as source, (metadata / "images.jsonl").open("w") as target:
        for line in source:
            row = json.loads(line)
            vid = Path(row["source_video"]).stem
            if ((row["split"] == "train" and vid in EXCLUDE_TRAIN)
                    or (row["split"] == "val" and vid in EXCLUDE_VAL)):
                removed_images[vid] += 1
                continue
            target.write(line)
            image_count += 1
    if (seen["train"] != set(split["train"]) or seen["val20"] != set(split["val20"])
            or seen["new30"] != set(split["new30"])):
        raise RuntimeError("Manifest source membership differs from clean split")
    if dict(removed_windows) != {"vid_mtxwk77pba622d": 326, "vid_mu31qed1aaf075": 167,
                                 "vid_mtyozrerb65868": 321, "vid_mtzaxwm0617fe8": 164}:
        raise RuntimeError(f"Unexpected removed window counts: {removed_windows}")
    if any(removed_images[vid] != 4 * count for vid, count in removed_windows.items()):
        raise RuntimeError("Removed image counts differ from four-frame windows")
    if image_count != 4 * sum(cohort_counts.values()):
        raise RuntimeError("Image and window counts disagree")
    summary = dict(base_summary)
    summary.update({
        "version": "v8_val48_trainclean_overlay_v1", "dataset_root": str(FINAL),
        "split_manifest": str(FINAL / "metadata/split.json"),
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
        "training_view": "manifest-only temporal overlay; train/val image folders link to the verified r2 bytes",
        "online_selection": "clean val48 F1; full frozen val20 and val50 scored separately from r2 after training",
        "excluded_train_reencodes": sorted(EXCLUDE_TRAIN),
        "excluded_validation_reencodes": sorted(EXCLUDE_VAL),
        "frozen_val20": {"source_dataset": str(BASE), "source_count": len(seen["val20"]),
                         "window_count": cohort_counts["val20"],
                         "note": "Clean val18 subset; full frozen val20 remains in r2"},
        "validation_leak_audit": str(WORK / "v8_validation_leak_audit.json"),
        "initializer_caveat": "The stage-two legacy initializer already trained on one historical train/val re-encode; clean val18/new30 and full10 are primary comparisons.",
    })
    (metadata / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (BUILDING / "README.txt").write_text(
        "Manifest-only four-frame training overlay. The train/val image folders are symlinks "
        "to r2 and contain two excluded training and two excluded validation videos; only metadata/windows.jsonl and "
        "metadata/images.jsonl define the clean training population. Do not use folder scans "
        "or ImageFolder on this overlay. See metadata/summary.json and the leak audit.\n"
    )
    BUILDING.rename(FINAL)
    ready = {"status": "ready", "dataset": str(FINAL),
             "window_count": summary["window_count"], "image_count": image_count,
             "train_sources": len(seen["train"]), "val_sources": len(seen["val20"]) + len(seen["new30"]),
             "excluded_train_reencodes": sorted(EXCLUDE_TRAIN),
             "excluded_validation_reencodes": sorted(EXCLUDE_VAL),
             "windows_sha256": summary["windows_sha256"],
             "summary_sha256": sha256(FINAL / "metadata/summary.json"),
             "validation_leak_audit": str(WORK / "v8_validation_leak_audit.json")}
    (WORK / "val48_trainclean_dataset_ready.json").write_text(json.dumps(ready, indent=2) + "\n")
    print(json.dumps(ready))


if __name__ == "__main__":
    main()
