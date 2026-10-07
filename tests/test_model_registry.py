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
    assert gate["outside_3s_action_fp_rejected"] == 94
    assert gate["outside_3s_action_fp_remaining"] == 20
    assert gate["outside_3s_action_fp_rejected_fraction"] == 94 / 114
    assert Path(gate["impact_report"]).is_absolute()
    assert Path(gate["validation_result"]).is_absolute()


def test_best_model_records_portable_runtime_assets() -> None:
    """The Qwen integration model has checksum-bound standalone weights."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    runtime = registry["best_model"]["runtime"]

    assert runtime["backend"] == "pytorch_qwen"
    assert runtime["onnx_status"] == "not_exported_pytorch_runtime_selected"
    assert runtime["tensorrt_compatible"] is False
    assert runtime["input_shape"] == ["batch", 4, 3, 384, 672]
    assert Path(runtime["weights"]).is_absolute()
    assert len(runtime["weights_sha256"]) == 64
    assert Path(runtime["parity_evidence"]).is_absolute()


def test_beach_domain_model_records_production_selection_and_evidence() -> None:
    """Beach routing should resolve the explicitly approved v7 specialist."""
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    model = registry["domain_models"]["beach"]

    assert model["status"] == "production"
    assert model["production"] is True
    assert model["production_scope"] == "beach_volleyball"
    assert model["pipeline_status"] == "production_model_not_yet_integrated"
    assert model["threshold"] == 0.5
    assert model["f1"] == 0.9336538461538463
    assert model["validation"]["validation_videos"] == 9
    assert model["validation"]["validation_windows"] == 1991
    assert model["full_video_gameplay_gate"]["status"] == "not_run_for_beach_domain"
    assert model["publication_status"] == "published_production"
    assert (
        model["publication"]["remote_path"]
        == "googdrive:skyball/checkpoints/gameplay/"
        "v7_beach_siglip512_prod_finetune_gpu0_20260723"
    )
    assert (
        model["publication"]["catalog_pointer"]
        == "recommended_by_domain.beach_volleyball.gameplay"
    )
    assert len(model["publication"]["manifest_sha256"]) == 64
    assert Path(model["checkpoint"]).is_absolute()
    assert len(model["checkpoint_sha256"]) == 64
