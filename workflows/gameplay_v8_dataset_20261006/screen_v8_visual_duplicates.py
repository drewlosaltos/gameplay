#!/usr/bin/env python3
"""Screen train/validation raw-video re-encodes using sparse extracted frames."""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")
OUT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006/visual_duplicate_screen.json")
QUANTILES = (0.15, 0.50, 0.85)


def phash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot decode extracted frame: {path}")
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    frequencies = cv2.dct(small)[:8, :8].flatten()
    median = float(np.median(frequencies[1:]))
    result = 0
    for frequency in frequencies[1:]:
        result = (result << 1) | int(frequency > median)
    return result


def main() -> None:
    summary = json.loads((DATA / "metadata/summary.json").read_text())
    cohort = {name: label for label, names in summary["source_videos_by_cohort"].items()
              for name in names}
    windows = defaultdict(list)
    with (DATA / "metadata/windows.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            windows[row["video_id"]].append((float(row["start_sec"]), row["frame_paths"][0]))
    if set(windows) != set(cohort):
        raise RuntimeError("Window manifest and published cohort IDs differ")
    selected = {}
    for vid, records in windows.items():
        records.sort()
        selected[vid] = [records[round((len(records) - 1) * fraction)][1]
                         for fraction in QUANTILES]
    signatures = {vid: [phash(DATA / relative) for relative in paths]
                  for vid, paths in selected.items()}
    train = [vid for vid in signatures if cohort[vid] == "train"]
    validation = [vid for vid in signatures if cohort[vid] in {"val20", "new30"}]
    comparisons = []
    for val in validation:
        for trn in train:
            distances = [(a ^ b).bit_count() for a, b in zip(signatures[val], signatures[trn], strict=True)]
            comparisons.append((sum(distances), max(distances), val, trn, distances))
    comparisons.sort()
    closest = []
    for total, maximum, val, trn, distances in comparisons[:30]:
        closest.append({"validation_video": val, "validation_cohort": cohort[val],
                        "train_video": trn, "hamming_distances": distances,
                        "total_hamming": total, "maximum_hamming": maximum,
                        "validation_frames": selected[val], "train_frames": selected[trn]})
    high_similarity = [item for item in closest
                       if item["maximum_hamming"] <= 10 and item["total_hamming"] <= 24]
    report = {"status": "screen_complete", "scope": "Three approximate timeline-quantile pHashes per video; "
              "a screen for obvious aligned re-encodes, not proof against shifted/cropped duplicates",
              "dataset": str(DATA), "videos": len(signatures), "train_videos": len(train),
              "validation_videos": len(validation), "pairs_compared": len(comparisons),
              "quantiles": QUANTILES, "high_similarity_threshold": "max frame distance <=10 and sum <=24",
              "high_similarity_pairs": high_similarity, "closest_pairs": closest}
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(OUT)
    print(json.dumps({"status": report["status"], "videos": len(signatures),
                      "pairs_compared": len(comparisons),
                      "high_similarity_pairs": len(high_similarity),
                      "closest_total_hamming": closest[0]["total_hamming"]}))


if __name__ == "__main__":
    main()
