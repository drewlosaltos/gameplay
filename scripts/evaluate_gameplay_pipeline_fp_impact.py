#!/usr/bin/env python3
"""Retrospectively measure gameplay gating impact on pipeline false positives.

This standalone analysis reads completed pipeline artifacts but does not modify
or rerun the pipeline. Action false positives are the unmatched predictions in
``action_match_audit.json``. Play false positives are the predicted plays that
the suite marks invalid because they are not near any confirmed GT event.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evaluate_gameplay_gate import DEFAULT_EVALUATION_ROOT
from evaluate_gameplay_gate import DEFAULT_OUTPUT_ROOT
from evaluate_gameplay_gate import Interval
from evaluate_gameplay_gate import MatchData
from evaluate_gameplay_gate import Policy
from evaluate_gameplay_gate import aggregate_metrics
from evaluate_gameplay_gate import atomic_write_json
from evaluate_gameplay_gate import evaluate_policy
from evaluate_gameplay_gate import load_match_data
from evaluate_gameplay_gate import point_is_covered
from evaluate_gameplay_gate import predicted_intervals


DEFAULT_PIPELINE_ROOT = Path(
    "/mnt/t/output/skyball_pipeline/"
    "evaluation_suite_serve_primary_e2e_cached_detector_v1_20260717"
)
DEFAULT_IMPACT_OUTPUT = (
    DEFAULT_OUTPUT_ROOT / "pipeline_fp_impact_quality_full_recall_v1"
)


def intervals_overlap(intervals: list[Interval], start_sec: float, end_sec: float) -> bool:
    """Return whether a closed prediction span overlaps a gameplay interval."""
    return any(
        interval.start_sec <= end_sec and start_sec <= interval.end_sec
        for interval in intervals
    )


def load_scores(study_root: Path, match: MatchData) -> tuple[list[float], list[float]]:
    """Load cached starts and scores for one match."""
    starts = []
    scores = []
    with (study_root / match.match_id / "scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            starts.append(float(row["window_start_sec"]))
            scores.append(float(row["score_gameplay"]))
    return starts, scores


def predicted_gameplay_intervals(
    study_root: Path,
    match: MatchData,
    policy: Policy,
) -> list[Interval]:
    """Build predicted gameplay intervals for one cached match."""
    starts, scores = load_scores(study_root, match)
    return predicted_intervals(
        starts_sec=starts,
        scores=scores,
        threshold=policy.threshold,
        window_sec=match.window_sec,
        merge_gap_sec=policy.merge_gap_sec,
        pad_before_sec=policy.pad_before_sec,
        pad_after_sec=policy.pad_after_sec,
        video_duration_sec=match.duration_sec,
    )


def analyze_match(
    match: MatchData,
    study_root: Path,
    pipeline_root: Path,
    suite_summary: dict[str, Any],
    policy: Policy,
) -> dict[str, Any]:
    """Measure retrospective action and play rejection for one match."""
    intervals = predicted_gameplay_intervals(study_root, match, policy)
    audit_path = pipeline_root / match.match_id / "action_match_audit.json"
    plays_path = pipeline_root / match.match_id / "plays.csv"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    fps = float(audit["fps"])

    action_false_positives = audit["unmatched_predictions"]
    event_times_sec = [
        event_time
        for rally in match.rallies
        for event_time in rally.event_times_sec
    ]
    outside_event_margin_false_positives = [
        row
        for row in action_false_positives
        if all(
            abs(float(row["pred"]["fnum"]) / fps - event_time) > 3.0
            for event_time in event_times_sec
        )
    ]
    rejected_action_false_positives = [
        {
            **row,
            "time_sec": float(row["pred"]["fnum"]) / fps,
        }
        for row in action_false_positives
        if not point_is_covered(intervals, float(row["pred"]["fnum"]) / fps)
    ]
    rejected_outside_event_margin_false_positives = [
        {
            **row,
            "time_sec": float(row["pred"]["fnum"]) / fps,
        }
        for row in outside_event_margin_false_positives
        if not point_is_covered(intervals, float(row["pred"]["fnum"]) / fps)
    ]
    matched_actions = audit["matches"]
    rejected_matched_actions = [
        {
            **row,
            "time_sec": float(row["pred"]["fnum"]) / fps,
        }
        for row in matched_actions
        if not point_is_covered(intervals, float(row["pred"]["fnum"]) / fps)
    ]

    invalid_play_details = suite_summary[match.match_id]["gameplay"][
        "invalid_play_details"
    ]
    invalid_play_numbers = {int(row["pnum"]) for row in invalid_play_details}
    with plays_path.open(newline="", encoding="utf-8") as handle:
        plays = list(csv.DictReader(handle))

    def play_is_retained(row: dict[str, str]) -> bool:
        """Return whether an existing predicted play overlaps the gate."""
        return intervals_overlap(
            intervals,
            float(row["fnum-start"]) / fps,
            float(row["fnum-last"]) / fps,
        )

    invalid_plays = [row for row in plays if int(row["pnum"]) in invalid_play_numbers]
    valid_plays = [row for row in plays if int(row["pnum"]) not in invalid_play_numbers]
    rejected_invalid_plays = [row for row in invalid_plays if not play_is_retained(row)]
    rejected_valid_plays = [row for row in valid_plays if not play_is_retained(row)]
    gameplay_metrics = evaluate_policy(match, policy)
    return {
        "match_id": match.match_id,
        "fps": fps,
        "policy": asdict(policy),
        "gameplay": gameplay_metrics,
        "action_false_positives": {
            "current": len(action_false_positives),
            "rejected": len(rejected_action_false_positives),
            "remaining": len(action_false_positives)
            - len(rejected_action_false_positives),
            "rejected_fraction": (
                len(rejected_action_false_positives) / len(action_false_positives)
                if action_false_positives
                else 0.0
            ),
            "rejected_details": rejected_action_false_positives,
        },
        "outside_gt_event_margin_action_false_positives": {
            "definition": (
                "unmatched predicted action more than 3 seconds from every "
                "confirmed synchronized GT action"
            ),
            "current": len(outside_event_margin_false_positives),
            "rejected": len(rejected_outside_event_margin_false_positives),
            "remaining": len(outside_event_margin_false_positives)
            - len(rejected_outside_event_margin_false_positives),
            "rejected_fraction": (
                len(rejected_outside_event_margin_false_positives)
                / len(outside_event_margin_false_positives)
                if outside_event_margin_false_positives
                else 0.0
            ),
            "rejected_details": rejected_outside_event_margin_false_positives,
        },
        "unmatched_predictions_within_gt_event_margin": len(action_false_positives)
        - len(outside_event_margin_false_positives),
        "matched_true_actions": {
            "current": len(matched_actions),
            "rejected": len(rejected_matched_actions),
            "remaining": len(matched_actions) - len(rejected_matched_actions),
            "rejected_details": rejected_matched_actions,
        },
        "invalid_predicted_plays": {
            "current": len(invalid_plays),
            "rejected": len(rejected_invalid_plays),
            "remaining": len(invalid_plays) - len(rejected_invalid_plays),
            "rejected_fraction": (
                len(rejected_invalid_plays) / len(invalid_plays)
                if invalid_plays
                else 0.0
            ),
            "rejected_details": rejected_invalid_plays,
        },
        "valid_predicted_plays": {
            "current": len(valid_plays),
            "rejected": len(rejected_valid_plays),
            "remaining": len(valid_plays) - len(rejected_valid_plays),
            "rejected_details": rejected_valid_plays,
        },
    }


def aggregate_impact(per_match: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate false-positive and gameplay metrics over all matches."""

    def sum_nested(section: str, field: str) -> int:
        """Sum an integer field from one impact section."""
        return sum(int(row[section][field]) for row in per_match)

    action_fp_current = sum_nested("action_false_positives", "current")
    action_fp_rejected = sum_nested("action_false_positives", "rejected")
    outside_fp_current = sum_nested(
        "outside_gt_event_margin_action_false_positives", "current"
    )
    outside_fp_rejected = sum_nested(
        "outside_gt_event_margin_action_false_positives", "rejected"
    )
    invalid_play_current = sum_nested("invalid_predicted_plays", "current")
    invalid_play_rejected = sum_nested("invalid_predicted_plays", "rejected")
    return {
        "gameplay": aggregate_metrics(row["gameplay"] for row in per_match),
        "action_false_positives": {
            "current": action_fp_current,
            "rejected": action_fp_rejected,
            "remaining": action_fp_current - action_fp_rejected,
            "rejected_fraction": (
                action_fp_rejected / action_fp_current if action_fp_current else 0.0
            ),
        },
        "outside_gt_event_margin_action_false_positives": {
            "definition": (
                "unmatched predicted action more than 3 seconds from every "
                "confirmed synchronized GT action"
            ),
            "current": outside_fp_current,
            "rejected": outside_fp_rejected,
            "remaining": outside_fp_current - outside_fp_rejected,
            "rejected_fraction": (
                outside_fp_rejected / outside_fp_current if outside_fp_current else 0.0
            ),
        },
        "unmatched_predictions_within_gt_event_margin": sum(
            int(row["unmatched_predictions_within_gt_event_margin"])
            for row in per_match
        ),
        "matched_true_actions": {
            "current": sum_nested("matched_true_actions", "current"),
            "rejected": sum_nested("matched_true_actions", "rejected"),
            "remaining": sum_nested("matched_true_actions", "remaining"),
        },
        "invalid_predicted_plays": {
            "current": invalid_play_current,
            "rejected": invalid_play_rejected,
            "remaining": invalid_play_current - invalid_play_rejected,
            "rejected_fraction": (
                invalid_play_rejected / invalid_play_current
                if invalid_play_current
                else 0.0
            ),
        },
        "valid_predicted_plays": {
            "current": sum_nested("valid_predicted_plays", "current"),
            "rejected": sum_nested("valid_predicted_plays", "rejected"),
            "remaining": sum_nested("valid_predicted_plays", "remaining"),
        },
    }


def write_csv(path: Path, per_match: list[dict[str, Any]]) -> None:
    """Write the requested per-match comparison table."""
    fieldnames = (
        "match_id",
        "true_gameplay_coverage_percent",
        "event_coverage_percent",
        "fully_covered_rallies_percent",
        "current_action_fp",
        "action_fp_rejected",
        "action_fp_remaining",
        "current_outside_3s_action_fp",
        "outside_3s_action_fp_rejected",
        "outside_3s_action_fp_remaining",
        "unmatched_predictions_within_3s",
        "matched_true_actions_rejected",
        "current_invalid_plays",
        "invalid_plays_rejected",
        "invalid_plays_remaining",
        "valid_plays_rejected",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_match:
            writer.writerow(
                {
                    "match_id": row["match_id"],
                    "true_gameplay_coverage_percent": 100
                    * float(row["gameplay"]["annotated_gameplay_recall"]),
                    "event_coverage_percent": 100
                    * float(row["gameplay"]["event_recall"]),
                    "fully_covered_rallies_percent": 100
                    * float(row["gameplay"]["rally_full_coverage_recall"]),
                    "current_action_fp": row["action_false_positives"]["current"],
                    "action_fp_rejected": row["action_false_positives"]["rejected"],
                    "action_fp_remaining": row["action_false_positives"]["remaining"],
                    "current_outside_3s_action_fp": row[
                        "outside_gt_event_margin_action_false_positives"
                    ]["current"],
                    "outside_3s_action_fp_rejected": row[
                        "outside_gt_event_margin_action_false_positives"
                    ]["rejected"],
                    "outside_3s_action_fp_remaining": row[
                        "outside_gt_event_margin_action_false_positives"
                    ]["remaining"],
                    "unmatched_predictions_within_3s": row[
                        "unmatched_predictions_within_gt_event_margin"
                    ],
                    "matched_true_actions_rejected": row["matched_true_actions"][
                        "rejected"
                    ],
                    "current_invalid_plays": row["invalid_predicted_plays"]["current"],
                    "invalid_plays_rejected": row["invalid_predicted_plays"]["rejected"],
                    "invalid_plays_remaining": row["invalid_predicted_plays"]["remaining"],
                    "valid_plays_rejected": row["valid_predicted_plays"]["rejected"],
                }
            )


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    """Write a concise human-readable comparison report."""
    aggregate = report["aggregate"]
    lines = [
        "# Gameplay Gate Impact on Current Pipeline False Positives",
        "",
        "This is a retrospective standalone analysis; the pipeline was not rerun or modified.",
        "",
        f"Pipeline baseline: `{report['pipeline_root']}`",
        "",
        f"Policy: `{json.dumps(report['policy'], sort_keys=True)}`",
        "",
        f"Selection caveat: {report['selection_caveat']}",
        "",
        "| Match | gameplay coverage | current FP outside ±3s | outside FP rejected | outside FP remaining | all unmatched predictions | matched true actions rejected | invalid plays | invalid plays rejected |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["per_match"]:
        lines.append(
            f"| {row['match_id']} | "
            f"{100 * float(row['gameplay']['annotated_gameplay_recall']):.4f}% | "
            f"{row['outside_gt_event_margin_action_false_positives']['current']} | "
            f"{row['outside_gt_event_margin_action_false_positives']['rejected']} | "
            f"{row['outside_gt_event_margin_action_false_positives']['remaining']} | "
            f"{row['action_false_positives']['current']} | "
            f"{row['matched_true_actions']['rejected']} | "
            f"{row['invalid_predicted_plays']['current']} | "
            f"{row['invalid_predicted_plays']['rejected']} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Annotated gameplay coverage: {100 * float(aggregate['gameplay']['annotated_gameplay_recall']):.4f}%",
            f"- Target action FP rejected: {aggregate['outside_gt_event_margin_action_false_positives']['rejected']} / {aggregate['outside_gt_event_margin_action_false_positives']['current']} ({100 * float(aggregate['outside_gt_event_margin_action_false_positives']['rejected_fraction']):.2f}%)",
            f"- All unmatched predictions (secondary): {aggregate['action_false_positives']['current']}; within ±3 seconds: {aggregate['unmatched_predictions_within_gt_event_margin']}",
            f"- Matched true actions rejected: {aggregate['matched_true_actions']['rejected']} / {aggregate['matched_true_actions']['current']}",
            f"- Invalid plays rejected: {aggregate['invalid_predicted_plays']['rejected']} / {aggregate['invalid_predicted_plays']['current']} ({100 * float(aggregate['invalid_predicted_plays']['rejected_fraction']):.2f}%)",
            f"- Valid plays rejected: {aggregate['valid_predicted_plays']['rejected']} / {aggregate['valid_predicted_plays']['current']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run the retrospective quality-impact analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--study-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_IMPACT_OUTPUT)
    parser.add_argument("--gt-margin-sec", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--merge-gap-sec", type=float, default=90.0)
    parser.add_argument("--padding-sec", type=float, default=10.0)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = json.loads(
        (args.pipeline_root / "evaluation_suite_summary.json").read_text(
            encoding="utf-8"
        )
    )
    suite_summary = {
        row["match"]: row["summary"] for row in summary_payload["matches"]
    }
    policy = Policy(
        threshold=args.threshold,
        merge_gap_sec=args.merge_gap_sec,
        pad_before_sec=args.padding_sec,
        pad_after_sec=args.padding_sec,
    )
    matches = load_match_data(args.evaluation_root, args.study_root, args.gt_margin_sec)
    per_match = [
        analyze_match(
            match=match,
            study_root=args.study_root,
            pipeline_root=args.pipeline_root,
            suite_summary=suite_summary,
            policy=policy,
        )
        for match in matches
    ]
    report = {
        "schema_version": 1,
        "scope": "standalone retrospective quality-impact analysis",
        "pipeline_modified": False,
        "pipeline_root": str(args.pipeline_root),
        "study_root": str(args.study_root),
        "policy": asdict(policy),
        "selection_caveat": (
            "policy selected on these same ten evaluation matches to maximize current "
            "pipeline FP rejection subject to complete annotated gameplay/event/rally "
            "coverage and zero rejected matched actions or valid plays"
        ),
        "per_match": per_match,
        "aggregate": aggregate_impact(per_match),
    }
    atomic_write_json(args.output_dir / "pipeline_fp_impact.json", report)
    write_csv(args.output_dir / "pipeline_fp_impact.csv", per_match)
    write_markdown(args.output_dir / "pipeline_fp_impact.md", report)
    print(json.dumps(report["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
