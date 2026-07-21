"""Tests for the standalone gameplay-gating feasibility evaluator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import yaml

from skyball_gameplay import modeling


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_gameplay_gate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gameplay_gate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


Interval = MODULE.Interval


class DummyFeatureEncoder(torch.nn.Module):
    """Small timm-like encoder used to exercise checkpoint reconstruction."""

    num_features = 3

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return one three-value feature vector per image."""
        return inputs.mean(dim=(2, 3))


def test_defaults_follow_recommended_registry_model() -> None:
    """Default scoring must not silently fall back to an obsolete checkpoint."""
    registry = yaml.safe_load(
        (Path(__file__).parents[1] / "model_registry.yaml").read_text(
            encoding="utf-8"
        )
    )["best_model"]

    assert MODULE.DEFAULT_CHECKPOINT == Path(registry["checkpoint"])
    assert MODULE.DEFAULT_REGISTRY_THRESHOLD == registry["threshold"]
    assert registry["model_id"] in str(MODULE.DEFAULT_OUTPUT_ROOT)


def test_merge_intervals_fills_requested_gap() -> None:
    """Intervals separated by at most the requested gap should merge."""
    merged = MODULE.merge_intervals(
        [Interval(1.0, 3.0), Interval(5.0, 7.0), Interval(12.0, 13.0)],
        max_gap_sec=2.0,
    )

    assert merged == [Interval(1.0, 7.0), Interval(12.0, 13.0)]


def test_predicted_intervals_apply_window_extent_gap_and_padding() -> None:
    """Positive windows should become padded, clipped coarse intervals."""
    predicted = MODULE.predicted_intervals(
        starts_sec=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        scores=[0.1, 0.8, 0.2, 0.2, 0.9, 0.1],
        threshold=0.5,
        window_sec=4.0,
        merge_gap_sec=0.0,
        pad_before_sec=2.0,
        pad_after_sec=2.0,
        video_duration_sec=10.0,
    )

    assert predicted == [Interval(0.0, 10.0)]


def test_coverage_metrics_reports_time_event_and_full_rally_recall() -> None:
    """Coverage metrics should distinguish any-event from full-rally recall."""
    ground_truth = [
        MODULE.GroundTruthRally("1", "1", Interval(2.0, 5.0), (2.0, 4.0)),
        MODULE.GroundTruthRally("1", "2", Interval(8.0, 10.0), (8.0, 10.0)),
    ]
    predicted = [Interval(1.0, 4.5), Interval(8.0, 10.0)]

    metrics = MODULE.coverage_metrics(
        predicted=predicted,
        rallies=ground_truth,
        video_duration_sec=12.0,
    )

    assert metrics["annotated_gameplay_seconds"] == 5.0
    assert metrics["covered_gameplay_seconds"] == 4.5
    assert metrics["annotated_gameplay_recall"] == 0.9
    assert metrics["event_recall"] == 1.0
    assert metrics["fully_covered_rallies"] == 1
    assert metrics["rally_full_coverage_recall"] == 0.5
    assert metrics["retained_total_seconds"] == 5.5
    assert metrics["retained_annotation_negative_seconds"] == 1.0
    assert metrics["excluded_video_seconds"] == 6.5


def test_union_intersection_avoids_double_counting_overlap() -> None:
    """Overlapping intervals must not inflate duration or coverage."""
    left = [Interval(0.0, 5.0), Interval(4.0, 8.0)]
    right = [Interval(2.0, 6.0)]

    assert MODULE.interval_union_duration(left) == 8.0
    assert MODULE.interval_intersection_duration(left, right) == 4.0


def test_load_model_reconstructs_timm_temporal_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Full-video scoring should honor timm architecture and normalization metadata."""
    source = modeling.TemporalFeatureAveragePoolClassifier(
        DummyFeatureEncoder(),
        feature_dim=3,
        num_classes=2,
        dropout=0.1,
    )
    checkpoint_path = tmp_path / "siglip.pth.tar"
    torch.save(
        {
            "arch": "resnet18",
            "timm_model": "dummy_siglip",
            "input_frames": 4,
            "classes": ["non_gameplay", "gameplay"],
            "num_classes": 2,
            "image_size": [512, 512],
            "normalization_mean": (0.5, 0.5, 0.5),
            "normalization_std": (0.5, 0.5, 0.5),
            "temporal_dropout": 0.1,
            "epoch": 3,
            "state_dict": {
                f"module.{key}": value for key, value in source.state_dict().items()
            },
        },
        checkpoint_path,
    )
    monkeypatch.setattr(
        modeling.timm,
        "create_model",
        lambda *_args, **_kwargs: DummyFeatureEncoder(),
    )

    model, metadata = MODULE.load_model(checkpoint_path, torch.device("cpu"))

    assert isinstance(model, modeling.TemporalFeatureAveragePoolClassifier)
    assert metadata["timm_model"] == "dummy_siglip"
    assert metadata["normalization_mean"] == [0.5, 0.5, 0.5]
    assert metadata["normalization_std"] == [0.5, 0.5, 0.5]
