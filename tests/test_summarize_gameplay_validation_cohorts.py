"""Tests for staged gameplay validation summaries."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "summarize_gameplay_validation_cohorts.py"
SPEC = importlib.util.spec_from_file_location("summarize_gameplay_validation_cohorts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_summarize_cohorts_uses_combined_threshold_for_each_slice() -> None:
    """Cohort reporting should share the threshold tuned on combined validation."""
    rows = [
        {
            "validation_cohort": "v2_fixed",
            "source_video": "legacy.mp4",
            "class_name": "gameplay",
            "target": 1,
            "score": 0.9,
        },
        {
            "validation_cohort": "v2_fixed",
            "source_video": "legacy.mp4",
            "class_name": "timeout",
            "target": 0,
            "score": 0.4,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "college.mp4",
            "class_name": "gameplay",
            "target": 1,
            "score": 0.6,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "college.mp4",
            "class_name": "setbreak",
            "target": 0,
            "score": 0.2,
        },
    ]

    summary = MODULE.summarize_cohorts(rows)

    assert summary["combined_best_f1_threshold"] == 0.6
    assert summary["combined"]["at_combined_threshold"]["f1"] == 1.0
    assert summary["v2_fixed"]["at_combined_threshold"]["f1"] == 1.0
    assert summary["v5_expanded"]["at_combined_threshold"]["f1"] == 1.0


def test_summarize_cohorts_reports_population_counts() -> None:
    """The summary should retain class counts for each validation population."""
    rows = [
        {
            "validation_cohort": "v2_fixed",
            "source_video": "legacy.mp4",
            "class_name": "gameplay",
            "target": 1,
            "score": 0.8,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "college.mp4",
            "class_name": "timeout",
            "target": 0,
            "score": 0.3,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "college.mp4",
            "class_name": "setbreak",
            "target": 0,
            "score": 0.7,
        },
    ]

    summary = MODULE.summarize_cohorts(rows)

    assert summary["v2_fixed"]["count"] == 1
    assert summary["v2_fixed"]["positive_count"] == 1
    assert summary["v5_expanded"]["count"] == 2
    assert summary["v5_expanded"]["negative_count"] == 2


def test_summarize_cohorts_reports_each_source_video() -> None:
    """Each held-out source should retain metrics at both shared thresholds."""
    rows = [
        {
            "validation_cohort": "v2_fixed",
            "source_video": "legacy.mp4",
            "class_name": "gameplay",
            "target": 1,
            "score": 0.9,
        },
        {
            "validation_cohort": "v2_fixed",
            "source_video": "legacy.mp4",
            "class_name": "timeout",
            "target": 0,
            "score": 0.2,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "college.mp4",
            "class_name": "gameplay",
            "target": 1,
            "score": 0.8,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "college.mp4",
            "class_name": "setbreak",
            "target": 0,
            "score": 0.1,
        },
    ]

    summary = MODULE.summarize_cohorts(rows)

    assert set(summary["per_source_video"]) == {"legacy.mp4", "college.mp4"}
    legacy = summary["per_source_video"]["legacy.mp4"]
    assert legacy["validation_cohort"] == "v2_fixed"
    assert legacy["at_combined_threshold"]["f1"] == 1.0


def test_summarize_cohorts_reports_confusion_by_original_class() -> None:
    """Original labels should expose predicted gameplay/non-gameplay counts."""
    rows = [
        {
            "validation_cohort": "v5_expanded",
            "source_video": "match.mp4",
            "class_name": "timeout",
            "target": 0,
            "score": 0.8,
        },
        {
            "validation_cohort": "v5_expanded",
            "source_video": "match.mp4",
            "class_name": "timeout",
            "target": 0,
            "score": 0.2,
        },
        {
            "validation_cohort": "v2_fixed",
            "source_video": "legacy.mp4",
            "class_name": "gameplay",
            "target": 1,
            "score": 0.9,
        },
    ]

    summary = MODULE.summarize_cohorts(rows)

    timeout = summary["confusion_by_original_class"]["timeout"]
    assert timeout["count"] == 2
    assert timeout["at_argmax_threshold"] == {
        "predicted_gameplay": 1,
        "predicted_non_gameplay": 1,
    }


def test_best_f1_threshold_does_not_recompute_full_metrics(
    monkeypatch,
) -> None:
    """Threshold selection should sweep sorted scores without quadratic rescoring."""
    calls = 0
    original = MODULE.binary_metrics

    def counted_binary_metrics(targets, scores, threshold):
        nonlocal calls
        calls += 1
        return original(targets, scores, threshold)

    monkeypatch.setattr(MODULE, "binary_metrics", counted_binary_metrics)
    rows = [
        {"target": 1, "score": 0.9},
        {"target": 0, "score": 0.8},
        {"target": 1, "score": 0.7},
        {"target": 0, "score": 0.1},
    ]

    assert MODULE.best_f1_threshold(rows) == 0.7
    assert calls == 0
