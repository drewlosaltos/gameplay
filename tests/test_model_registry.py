"""Tests for the canonical gameplay model registry."""

from __future__ import annotations

from pathlib import Path

import yaml


REGISTRY_PATH = Path(__file__).parents[1] / "model_registry.yaml"


def test_best_model_records_classifier_and_full_video_gate_metrics() -> None:
    """Research selection must preserve both classifier and pipeline-gate evidence."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    model = registry["best_model"]
    gate = model["full_video_gameplay_gate"]

    assert model["f1"] > 0
    assert gate["status"] == "pass"
    assert gate["annotated_gameplay_recall"] == 1.0
    assert gate["confirmed_gt_action_recall"] == 1.0
    assert gate["rally_full_coverage_recall"] == 1.0
    assert gate["matched_true_actions_rejected"] == 0
    assert gate["valid_predicted_plays_rejected"] == 0
    assert gate["outside_3s_action_fp_current"] == 114
    assert gate["outside_3s_action_fp_rejected"] == 92
    assert gate["outside_3s_action_fp_remaining"] == 22
    assert gate["outside_3s_action_fp_rejected_fraction"] == 92 / 114
    assert Path(gate["impact_report"]).is_absolute()
    assert Path(gate["validation_result"]).is_absolute()


def test_best_model_records_portable_runtime_assets() -> None:
    """The recommended integration model should resolve its ONNX handoff asset."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    runtime = registry["best_model"]["runtime"]

    assert runtime["onnx_status"] == "validated"
    assert runtime["tensorrt_compatible"] is True
    assert runtime["input_shape"] == ["batch", 4, 3, 512, 512]
    assert Path(runtime["onnx"]).is_absolute()
    assert Path(runtime["onnx_parity_evidence"]).is_absolute()
