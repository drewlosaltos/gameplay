#!/usr/bin/env python3
"""Build an immutable gameplay publication manifest from validated evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return a file's SHA-256 digest.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object.

    Args:
        path: JSON file to read.

    Returns:
        Parsed JSON object.
    """
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def git_output(repo: Path, *args: str) -> str:
    """Return one Git command's stripped output.

    Args:
        repo: Git working tree.
        *args: Arguments following ``git``.

    Returns:
        Stripped standard output.
    """
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def artifact_role(relative_path: str) -> str:
    """Map a portable bundle path to its manifest role.

    Args:
        relative_path: POSIX path relative to the bundle root.

    Returns:
        Stable semantic role.
    """
    exact = {
        "README.md": "model_card",
        "model.onnx": "portable_onnx",
        "model.onnx.json": "onnx_runtime_contract",
        "gameplay_operating_points.json": "classifier_operating_points",
        "evidence/onnx_parity.json": "pytorch_onnx_real_window_parity",
        "evidence/tensorrt_build.json": "tensorrt_build_evidence",
        "evidence/tensorrt_parity_v2.json": "pytorch_tensorrt_real_window_parity",
        "evidence/classifier/metrics.json": "combined_validation_metrics",
        "evidence/classifier/validation_cohort_metrics.json": "validation_cohort_metrics",
        "evidence/classifier/predictions.csv": "validation_prediction_scores",
        "evidence/classifier/dataset_validation.json": "dataset_integrity_evidence",
        "evidence/full_video_gate/pipeline_fp_impact.json": "full_video_pipeline_fp_impact",
        "evidence/full_video_gate/pipeline_fp_impact.csv": "full_video_pipeline_fp_table",
        "evidence/full_video_gate/pipeline_fp_impact.md": "full_video_pipeline_fp_summary",
        "evidence/full_video_gate/policy_tuning_coarse_v1.json": "full_video_policy_search",
        "evidence/full_video_gate/gameplay_gate_eval10_v2.json": "full_video_gate_contract",
        "evidence/full_video_gate/validation_gate_result_v2.json": "full_video_gate_result",
        "evidence/full_video_gate/validation_gate_result_v2.md": "full_video_gate_result_summary",
    }
    if relative_path in exact:
        return exact[relative_path]
    if relative_path.endswith(".pth.tar"):
        return "pytorch_checkpoint"
    if relative_path.startswith("evidence/full_video_scores/"):
        return (
            "full_video_score_metadata"
            if relative_path.endswith("score_metadata.json")
            else "full_video_window_scores"
        )
    raise ValueError(f"No artifact role for {relative_path}")


def parse_args() -> argparse.Namespace:
    """Parse manifest-builder arguments.

    Returns:
        Parsed bundle settings.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).parents[1])
    parser.add_argument("--published-at", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate bundle evidence and write ``artifact.json``."""
    args = parse_args()
    manifest_path = args.bundle / "artifact.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Refusing existing manifest: {manifest_path}")
    if args.bundle.name != args.model_id:
        raise ValueError("Bundle directory name must match model-id")

    runtime = load_json(args.bundle / "model.onnx.json")
    cohort = load_json(
        args.bundle / "evidence/classifier/validation_cohort_metrics.json"
    )
    impact = load_json(
        args.bundle / "evidence/full_video_gate/pipeline_fp_impact.json"
    )
    gate_result = load_json(
        args.bundle / "evidence/full_video_gate/validation_gate_result_v2.json"
    )
    onnx_parity = load_json(args.bundle / "evidence/onnx_parity.json")
    trt_build = load_json(args.bundle / "evidence/tensorrt_build.json")
    trt_parity = load_json(args.bundle / "evidence/tensorrt_parity_v2.json")
    if any(
        row.get("status") != "pass"
        for row in (gate_result, onnx_parity, trt_build, trt_parity)
    ):
        raise ValueError("Gate, ONNX, and TensorRT evidence must all pass")

    artifacts = []
    for path in sorted(args.bundle.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        relative_path = path.relative_to(args.bundle).as_posix()
        artifacts.append(
            {
                "path": relative_path,
                "role": artifact_role(relative_path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    combined = cohort["combined"]["at_combined_threshold"]
    gate = impact["aggregate"]
    target_fp = gate["outside_gt_event_margin_action_false_positives"]
    model = runtime["model"]
    git_status = git_output(args.repo, "status", "--porcelain=v1")
    published_at = args.published_at or datetime.now().astimezone().isoformat(
        timespec="seconds"
    )
    payload = {
        "schema_version": 1,
        "model_id": args.model_id,
        "model_kind": "gameplay_classification",
        "status": "recommended",
        "qualification_status": "validation_and_full_video_gate_qualified",
        "pipeline_status": "full_video_gate_passed_not_integrated",
        "production": False,
        "published_at": published_at,
        "architecture": {
            "family": "timm_siglip_base_temporal_average_pool",
            "backbone": model["timm_model"],
            "input_frames": model["input_frames"],
            "model_input_height": model["image_size"][0],
            "model_input_width": model["image_size"][1],
            "classes": model["classes"],
            "gameplay_class_index": model["gameplay_class_index"],
            "temporal_aggregation": "mean of per-frame encoder embeddings",
        },
        "preprocessing": runtime["runtime_contract"]["input"],
        "runtime": {
            "portable_model": "model.onnx",
            "input": runtime["runtime_contract"]["input"],
            "outputs": runtime["runtime_contract"]["outputs"],
            "temporal_sampling": runtime["runtime_contract"]["temporal_sampling"],
            "onnx_opset": runtime["runtime_contract"]["opset"],
            "onnx_parity_status": onnx_parity["status"],
            "onnx_gameplay_score_max_abs_error": onnx_parity["outputs"]["score_difference"]["max_abs"],
            "tensorrt_compatible": True,
            "tensorrt_version_validated": trt_build["tensorrt_version"],
            "tensorrt_profile_validated": trt_build["optimization_profile"],
            "tensorrt_parity_status": trt_parity["status"],
            "tensorrt_gameplay_score_max_abs_error": trt_parity["outputs"]["score_difference"]["max_abs"],
            "engine_publication": "not_published_machine_specific",
        },
        "operating_points": {
            "classifier": {
                "score": "softmax probability for gameplay class index 1",
                "best_f1_threshold": combined["threshold"],
                "artifact": "gameplay_operating_points.json",
            },
            "full_video_gate": {
                **impact["policy"],
                "selection": "maximize target FP rejection subject to all safety constraints",
            },
        },
        "validation": {
            "classifier": {
                "population": "gameplay_v5 combined validation",
                "count": combined["count"],
                "f1": combined["f1"],
                "precision": combined["precision"],
                "recall": combined["recall"],
                "balanced_accuracy": combined["balanced_accuracy"],
                "roc_auc": combined["roc_auc"],
                "threshold": combined["threshold"],
                "confusion": combined["confusion"],
            },
            "full_video_gate": {
                "contract_id": gate_result["contract_id"],
                "status": gate_result["status"],
                "matches": len(impact["per_match"]),
                "annotated_gameplay_recall": gate["gameplay"]["annotated_gameplay_recall"],
                "confirmed_gt_action_recall": gate["gameplay"]["event_recall"],
                "confirmed_gt_actions": gate["gameplay"]["events"],
                "rally_full_coverage_recall": gate["gameplay"]["rally_full_coverage_recall"],
                "rallies": gate["gameplay"]["rallies"],
                "target_fp_current": target_fp["current"],
                "target_fp_rejected": target_fp["rejected"],
                "target_fp_remaining": target_fp["remaining"],
                "target_fp_rejected_fraction": target_fp["rejected_fraction"],
                "matched_true_actions_rejected": gate["matched_true_actions"]["rejected"],
                "valid_predicted_plays_rejected": gate["valid_predicted_plays"]["rejected"],
                "selection_caveat": impact["selection_caveat"],
            },
        },
        "source": {
            "repository": "gameplay",
            "git_commit": git_output(args.repo, "rev-parse", "HEAD"),
            "dirty_tree": bool(git_status),
            "dirty_status_sha256": hashlib.sha256(git_status.encode()).hexdigest(),
            "checkpoint": runtime["checkpoint"],
            "checkpoint_sha256": runtime["checkpoint_sha256"],
            "checkpoint_epoch": model["checkpoint_epoch"],
            "dataset_version": "gameplay_v5",
        },
        "known_limitations": [
            "The temporal policy was selected on the same ten matches used for the retrospective gate report.",
            "Production integration and end-to-end pipeline release acceptance remain pending.",
            "TensorRT engines are runtime-specific and must be rebuilt from the published ONNX.",
        ],
        "artifacts": artifacts,
        "publication_note": (
            "Recommended training-side integration candidate. Publication does not "
            "enable gameplay filtering or constitute a production pipeline release."
        ),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
