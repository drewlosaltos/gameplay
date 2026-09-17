#!/usr/bin/env python3
"""Summarize one gameplay checkpoint across staged validation cohorts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    """Divide while returning zero for an empty denominator.

    Args:
        numerator: Value to divide.
        denominator: Divisor.

    Returns:
        Division result or zero.
    """
    return float(numerator / denominator) if denominator else 0.0


def roc_auc(targets: list[int], scores: list[float]) -> float | None:
    """Compute binary ROC AUC with average ranks for tied scores.

    Args:
        targets: Binary ground-truth labels.
        scores: Gameplay probabilities.

    Returns:
        ROC AUC, or ``None`` when only one class is present.
    """
    positive_count = sum(targets)
    negative_count = len(targets) - positive_count
    if not positive_count or not negative_count:
        return None
    ranked = sorted(zip(scores, targets), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positive_rank_sum += average_rank * sum(target for _, target in ranked[index:end])
        index = end
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def binary_metrics(targets: list[int], scores: list[float], threshold: float) -> dict:
    """Compute binary metrics at one gameplay threshold.

    Args:
        targets: Binary ground-truth labels.
        scores: Gameplay probabilities.
        threshold: Positive decision threshold.

    Returns:
        Accuracy, balanced accuracy, precision, recall, F1, AUC, and counts.
    """
    predictions = [int(score >= threshold) for score in scores]
    true_positive = sum(target == 1 and pred == 1 for target, pred in zip(targets, predictions))
    true_negative = sum(target == 0 and pred == 0 for target, pred in zip(targets, predictions))
    false_positive = sum(target == 0 and pred == 1 for target, pred in zip(targets, predictions))
    false_negative = sum(target == 1 and pred == 0 for target, pred in zip(targets, predictions))
    precision = safe_div(true_positive, true_positive + false_positive)
    recall = safe_div(true_positive, true_positive + false_negative)
    specificity = safe_div(true_negative, true_negative + false_positive)
    f1 = safe_div(2.0 * precision * recall, precision + recall)
    return {
        "threshold": threshold,
        "accuracy": safe_div(true_positive + true_negative, len(targets)),
        "balanced_accuracy": 0.5 * (recall + specificity),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc(targets, scores),
        "confusion": {
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
        },
        "count": len(targets),
        "positive_count": sum(targets),
        "negative_count": len(targets) - sum(targets),
    }


def best_f1_threshold(rows: list[dict]) -> float:
    """Select the combined-validation threshold with best F1.

    Args:
        rows: Records containing ``target`` and ``score``.

    Returns:
        Best threshold, using balanced accuracy as a tie-breaker.
    """
    ranked = sorted(
        ((float(row["score"]), int(row["target"])) for row in rows),
        reverse=True,
    )
    positive_count = sum(target for _, target in ranked)
    negative_count = len(ranked) - positive_count
    best_threshold = 0.5
    best_key = (-1.0, -1.0)
    true_positive = 0
    false_positive = 0
    ranked_index = 0
    thresholds = sorted({0.0, 1.0, *(score for score, _ in ranked)}, reverse=True)
    for threshold in thresholds:
        while ranked_index < len(ranked) and ranked[ranked_index][0] >= threshold:
            target = ranked[ranked_index][1]
            true_positive += target
            false_positive += 1 - target
            ranked_index += 1
        false_negative = positive_count - true_positive
        true_negative = negative_count - false_positive
        precision = safe_div(true_positive, true_positive + false_positive)
        recall = safe_div(true_positive, true_positive + false_negative)
        specificity = safe_div(true_negative, true_negative + false_positive)
        f1 = safe_div(2.0 * precision * recall, precision + recall)
        key = (f1, 0.5 * (recall + specificity))
        if key > best_key or (key == best_key and threshold < best_threshold):
            best_key = key
            best_threshold = threshold
    return best_threshold


def prediction_counts(rows: list[dict], threshold: float) -> dict[str, int]:
    """Count predicted gameplay and non-gameplay windows.

    Args:
        rows: Records containing gameplay scores.
        threshold: Positive gameplay decision threshold.

    Returns:
        Prediction counts for the two binary outcomes.
    """
    predicted_gameplay = sum(float(row["score"]) >= threshold for row in rows)
    return {
        "predicted_gameplay": predicted_gameplay,
        "predicted_non_gameplay": len(rows) - predicted_gameplay,
    }


def summarize_cohorts(rows: list[dict]) -> dict:
    """Summarize frozen, expanded, and combined validation populations.

    Args:
        rows: Validation predictions with cohort, target, and score fields.

    Returns:
        Cohort metrics using the threshold calibrated on combined validation.
    """
    if not rows:
        raise ValueError("No validation predictions were supplied")
    threshold = best_f1_threshold(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["validation_cohort"])].append(row)
    grouped["combined"] = rows
    summary: dict[str, object] = {"combined_best_f1_threshold": threshold}
    for cohort, cohort_rows in sorted(grouped.items()):
        targets = [int(row["target"]) for row in cohort_rows]
        scores = [float(row["score"]) for row in cohort_rows]
        summary[cohort] = {
            "count": len(cohort_rows),
            "positive_count": sum(targets),
            "negative_count": len(targets) - sum(targets),
            "at_argmax_threshold": binary_metrics(targets, scores, 0.5),
            "at_combined_threshold": binary_metrics(targets, scores, threshold),
        }

    source_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        source_groups[str(row.get("source_video") or "unknown")].append(row)
    summary["per_source_video"] = {}
    for source_video, source_rows in sorted(source_groups.items()):
        cohorts = {str(row["validation_cohort"]) for row in source_rows}
        if len(cohorts) != 1:
            raise ValueError(f"Source video occurs in multiple validation cohorts: {source_video}")
        targets = [int(row["target"]) for row in source_rows]
        scores = [float(row["score"]) for row in source_rows]
        summary["per_source_video"][source_video] = {
            "validation_cohort": next(iter(cohorts)),
            "at_argmax_threshold": binary_metrics(targets, scores, 0.5),
            "at_combined_threshold": binary_metrics(targets, scores, threshold),
        }

    class_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        class_name = str(row.get("original_category") or row.get("class_name") or "unknown")
        class_groups[class_name].append(row)
    summary["confusion_by_original_class"] = {
        class_name: {
            "count": len(class_rows),
            "positive_count": sum(int(row["target"]) for row in class_rows),
            "negative_count": len(class_rows) - sum(int(row["target"]) for row in class_rows),
            "at_argmax_threshold": prediction_counts(class_rows, 0.5),
            "at_combined_threshold": prediction_counts(class_rows, threshold),
        }
        for class_name, class_rows in sorted(class_groups.items())
    }
    return summary


def load_validation_records(dataset_root: Path) -> list[dict]:
    """Load validation windows in evaluator index order.

    Args:
        dataset_root: Extracted gameplay dataset root.

    Returns:
        Validation window records.
    """
    path = dataset_root / "metadata" / "windows.jsonl"
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") == "val":
                records.append(record)
    if not records:
        raise ValueError(f"No validation windows found in {path}")
    return records


def load_prediction_rows(dataset_root: Path, predictions: Path) -> list[dict]:
    """Join evaluator predictions to validation cohort metadata.

    Args:
        dataset_root: Extracted gameplay dataset root.
        predictions: Evaluator CSV produced with ``--save-predictions``.

    Returns:
        Cohort, target, and score records.
    """
    records = load_validation_records(dataset_root)
    rows = []
    with predictions.open("r", encoding="utf-8", newline="") as handle:
        for prediction in csv.DictReader(handle):
            index = int(prediction["index"])
            record = records[index]
            cohort = record.get("validation_cohort")
            if not isinstance(cohort, str) or not cohort:
                raise ValueError(f"Missing validation cohort for index {index}: {cohort!r}")
            if prediction.get("window_id") and prediction["window_id"] != record.get("window_id"):
                raise ValueError(f"Prediction/metadata window mismatch at index {index}")
            rows.append(
                {
                    "validation_cohort": cohort,
                    "target": int(prediction["binary_target"]),
                    "score": float(prediction["score_gameplay"]),
                    "source_video": record.get("source_video", "unknown"),
                    "class_name": record.get("class_name", "unknown"),
                    "original_category": record.get("original_category", ""),
                }
            )
    if len(rows) != len(records):
        raise ValueError(f"Expected {len(records)} predictions, found {len(rows)}")
    return rows


def main() -> None:
    """Write a staged-validation metrics JSON artifact."""
    args = parse_args()
    rows = load_prediction_rows(args.dataset_root, args.predictions)
    summary = summarize_cohorts(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
