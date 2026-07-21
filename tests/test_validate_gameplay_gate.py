"""Tests for the versioned gameplay validation gate."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_gameplay_gate.py"
SPEC = importlib.util.spec_from_file_location("validate_gameplay_gate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def sample_report() -> dict:
    """Return a minimal report matching the current ten-match population."""
    return {
        "per_match": [{} for _ in range(10)],
        "aggregate": {
            "gameplay": {
                "annotated_gameplay_recall": 1.0,
                "event_recall": 1.0,
                "events": 8848,
                "rally_full_coverage_recall": 1.0,
                "rallies": 1370,
            },
            "outside_gt_event_margin_action_false_positives": {
                "current": 114,
                "rejected": 92,
                "rejected_fraction": 92 / 114,
            },
            "matched_true_actions": {"current": 8318, "rejected": 0},
            "valid_predicted_plays": {"current": 1395, "rejected": 0},
        },
    }


def sample_contract() -> dict:
    """Return the joint safety/effectiveness contract used by the repository."""
    return {
        "schema_version": 1,
        "population_checks": [
            {"id": "matches", "path": "per_match", "operator": "length_eq", "value": 10},
            {"id": "events", "path": "aggregate.gameplay.events", "operator": "eq", "value": 8848},
            {"id": "rallies", "path": "aggregate.gameplay.rallies", "operator": "eq", "value": 1370},
            {
                "id": "target_fp_population",
                "path": "aggregate.outside_gt_event_margin_action_false_positives.current",
                "operator": "eq",
                "value": 114,
            },
            {
                "id": "matched_actions",
                "path": "aggregate.matched_true_actions.current",
                "operator": "eq",
                "value": 8318,
            },
            {
                "id": "valid_predicted_plays",
                "path": "aggregate.valid_predicted_plays.current",
                "operator": "eq",
                "value": 1395,
            },
        ],
        "metric_checks": [
            {
                "id": "gameplay_seconds",
                "label": "annotated gameplay seconds covered",
                "kind": "scalar",
                "path": "aggregate.gameplay.annotated_gameplay_recall",
                "operator": "min",
                "value": 1.0,
                "tolerance": 1e-9,
            },
            {
                "id": "confirmed_actions",
                "label": "confirmed GT actions covered",
                "kind": "scalar",
                "path": "aggregate.gameplay.event_recall",
                "operator": "min",
                "value": 1.0,
                "tolerance": 1e-9,
            },
            {
                "id": "rallies",
                "label": "rallies fully covered",
                "kind": "scalar",
                "path": "aggregate.gameplay.rally_full_coverage_recall",
                "operator": "min",
                "value": 1.0,
                "tolerance": 1e-9,
            },
            {
                "id": "outside_fp_rejection",
                "label": "outside-3-second action FP rejected",
                "kind": "ratio",
                "numerator_path": "aggregate.outside_gt_event_margin_action_false_positives.rejected",
                "denominator_path": "aggregate.outside_gt_event_margin_action_false_positives.current",
                "operator": "min",
                "value": 92 / 114,
                "minimum_numerator": 92,
                "tolerance": 1e-9,
            },
            {
                "id": "matched_action_safety",
                "label": "matched true actions rejected",
                "kind": "scalar",
                "path": "aggregate.matched_true_actions.rejected",
                "operator": "max",
                "value": 0,
                "tolerance": 0,
            },
            {
                "id": "valid_play_safety",
                "label": "valid predicted plays rejected",
                "kind": "scalar",
                "path": "aggregate.valid_predicted_plays.rejected",
                "operator": "max",
                "value": 0,
                "tolerance": 0,
            },
        ],
    }


def test_current_baseline_passes_gate() -> None:
    """The checked-in baseline measurements should pass all five checks."""
    result = MODULE.evaluate_contract(sample_report(), sample_contract())

    assert result["status"] == "pass"
    assert result["summary"] == {"metric_checks": 6, "passed": 6, "failed": 0}


def test_recall_regression_fails_gate() -> None:
    """Any missed annotated gameplay should fail the safety gate."""
    report = sample_report()
    report["aggregate"]["gameplay"]["annotated_gameplay_recall"] = 0.999

    result = MODULE.evaluate_contract(report, sample_contract())

    assert result["status"] == "fail"
    assert "gameplay_seconds" in result["failed_metric_ids"]


def test_fp_rejection_regression_fails_gate() -> None:
    """Rejecting fewer target false positives should fail non-regression."""
    report = sample_report()
    target = report["aggregate"]["outside_gt_event_margin_action_false_positives"]
    target["rejected"] = 91
    target["rejected_fraction"] = 91 / 114

    result = MODULE.evaluate_contract(report, sample_contract())

    assert result["status"] == "fail"
    assert "outside_fp_rejection" in result["failed_metric_ids"]


def test_rejected_matched_action_fails_gate() -> None:
    """Rejecting even one matched true action should fail the gate."""
    report = sample_report()
    report["aggregate"]["matched_true_actions"]["rejected"] = 1

    result = MODULE.evaluate_contract(report, sample_contract())

    assert result["status"] == "fail"
    assert "matched_action_safety" in result["failed_metric_ids"]


def test_rejected_valid_play_fails_gate() -> None:
    """Rejecting a currently valid play should fail the safety gate."""
    report = sample_report()
    report["aggregate"]["valid_predicted_plays"]["rejected"] = 1

    result = MODULE.evaluate_contract(report, sample_contract())

    assert result["status"] == "fail"
    assert "valid_play_safety" in result["failed_metric_ids"]


def test_population_mismatch_is_invalid_not_model_failure() -> None:
    """Changed evaluation membership should require a new versioned contract."""
    report = sample_report()
    report["aggregate"]["gameplay"]["events"] = 8847

    result = MODULE.evaluate_contract(report, sample_contract())

    assert result["status"] == "invalid_population"
    assert result["failed_population_ids"] == ["events"]
