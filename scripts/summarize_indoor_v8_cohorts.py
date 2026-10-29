#!/usr/bin/env python3
"""Split v8 val-50 predictions into frozen, new, and optional clean cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from skyball_gameplay.evaluation import binary_metrics


DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "validation/gameplay_indoor_v8_val50_v1.json"
SCORE_BIN_COUNT = 20


def score_histogram(targets: list[int], scores: list[float]) -> dict:
    """Fixed-width score distribution by true binary class."""
    counts = {"gameplay": [0] * SCORE_BIN_COUNT,
              "non_gameplay": [0] * SCORE_BIN_COUNT}
    for target, score in zip(targets, scores, strict=True):
        if target not in (0, 1) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Invalid binary target or gameplay score")
        name = "gameplay" if target else "non_gameplay"
        counts[name][min(int(score * SCORE_BIN_COUNT), SCORE_BIN_COUNT - 1)] += 1
    return {"bin_edges": [i / SCORE_BIN_COUNT for i in range(SCORE_BIN_COUNT + 1)],
            "counts_by_true_class": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--exclude-validation-sources", type=Path,
                        help="Confirmed re-encoded validation sources to omit from clean-cohort metrics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.split.read_text())
    val20 = set(manifest["val20"])
    new30 = set(manifest["new30"])
    if val20 & new30 or set(manifest["val50"]) != val20 | new30:
        raise ValueError("Invalid val-20/val-30 source membership")
    excluded = set()
    if args.exclude_validation_sources:
        audit = json.loads(args.exclude_validation_sources.read_text())
        if audit.get("status") != "confirmed_validation_leak":
            raise ValueError("Validation exclusion audit is not confirmed")
        excluded = {pair["validation_video"] for pair in audit["confirmed_reencoded_pairs"]}
        if len(excluded) != 2 or not excluded <= val20:
            raise ValueError("Expected exactly two confirmed frozen val20 exclusions")
    groups = {name: {"true": [], "pred": [], "score": [], "sources": set()}
              for name in (("val20", "new30", "val50", "clean_val18", "clean_val48")
                           if excluded else ("val20", "new30", "val50"))}
    with args.predictions.open(newline="") as handle:
        for row in csv.DictReader(handle):
            video_id = Path(row["source_video"]).stem
            if video_id in val20:
                cohort = "val20"
            elif video_id in new30:
                cohort = "new30"
            else:
                raise ValueError(f"Prediction outside v8 validation population: {video_id}")
            destinations = [cohort, "val50"]
            if excluded and video_id not in excluded:
                destinations.append("clean_val48")
                if cohort == "val20":
                    destinations.append("clean_val18")
            for name in destinations:
                group = groups[name]
                group["true"].append(int(row["binary_target"]))
                group["pred"].append(int(row["binary_prediction"]))
                group["score"].append(float(row["score_gameplay"]))
                group["sources"].add(video_id)
    if groups["val20"]["sources"] != val20 or groups["new30"]["sources"] != new30:
        raise RuntimeError("Prediction source membership incomplete")
    if excluded and (groups["clean_val18"]["sources"] != val20 - excluded
                     or groups["clean_val48"]["sources"] != (val20 - excluded) | new30):
        raise RuntimeError("Clean-cohort prediction source membership incomplete")
    result = {
        name: {"source_count": len(group["sources"]),
               "score_histogram": score_histogram(group["true"], group["score"]),
               **binary_metrics(group["true"], group["pred"], group["score"])}
        for name, group in groups.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
