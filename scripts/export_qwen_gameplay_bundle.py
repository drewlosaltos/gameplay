#!/usr/bin/env python3
"""Export a trained Qwen gameplay checkpoint as the pipeline's FP32 safetensors bundle.

Reusable form of workflows/qwen_v8_production_20261007/{export,finalize_bundle,smoke}.py.
The template bundle (the published prod bundle from Drive) supplies artifact.json's
architecture/runtime block and qwen_vision_config.json. The gate root is the output of the
full10 commands in REPRODUCE.md: <gate-root>/impact/{pipeline_fp_impact,validation_gate_result}.json
and <gate-root>/coverage/metrics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DROP = ("state_dict", "optimizer", "scheduler", "scaler")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def gate_summary(gate_root: Path) -> dict:
    impact = json.loads((gate_root / "impact/pipeline_fp_impact.json").read_text())
    gate = json.loads((gate_root / "impact/validation_gate_result.json").read_text())
    coverage = json.loads((gate_root / "coverage/metrics.json").read_text())
    cover = coverage["aggregate"]["candidate_selected_policy"]
    target = impact["aggregate"]["outside_gt_event_margin_action_false_positives"]
    return {
        "policy": impact["policy"], "gate_status": gate["status"],
        "confirmed_events": cover["events"], "confirmed_events_excluded": cover["events_excluded"],
        "rallies_fully_retained": cover["rallies_fully_preserved"],
        "target_fp_current": target["current"], "target_fp_rejected": target["rejected"],
        "outside_event_footage_percent_total_match": round(cover["outside_event_margin_admitted_pct_match"], 5),
    }


def rehash(out: Path) -> None:
    path = out / "artifact.json"
    artifact = json.loads(path.read_text())
    artifact["artifacts"] = [{"path": str(p.relative_to(out)), "sha256": sha256(p), "size_bytes": p.stat().st_size}
                             for p in sorted(out.rglob("*")) if p.is_file() and p != path]
    dump(path, artifact)


def export(args: argparse.Namespace) -> dict:
    import torch
    from safetensors.torch import save_file

    out = args.output
    checkpoint = args.checkpoint or next(args.run_dir.glob("best_*.pth.tar"))
    gate = gate_summary(args.gate_root)
    policy = {"threshold": args.threshold, "padding_sec_per_side": args.padding_sec,
              "merge_gap_sec": args.merge_gap_sec, "sample_fps": 1.0, "batch_size": 4}
    if (gate["policy"]["threshold"], gate["policy"]["pad_before_sec"], gate["policy"]["merge_gap_sec"]) != (
            args.threshold, args.padding_sec, args.merge_gap_sec):
        raise ValueError(f"Gate evidence used policy {gate['policy']}, not the requested operating point")
    if not args.allow_failed_gate and (gate["gate_status"] != "pass" or gate["confirmed_events_excluded"]):
        raise ValueError(f"Full10 gate did not pass: {gate}")
    out.mkdir(parents=True, exist_ok=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    tensors = {k.removeprefix("module."): v.contiguous() for k, v in state["state_dict"].items()}
    save_file(tensors, out / "model.safetensors")
    shutil.copy2(args.template_bundle / "qwen_vision_config.json", out)
    dump(out / "checkpoint_metadata.json", {k: v for k, v in state.items() if k not in DROP})
    config = args.run_dir / "config.json"
    shutil.copy2(config if config.is_file() else args.run_dir / "canonical_plan.json", out / "training_config.json")
    source = args.run_dir / "code_snapshot"
    if not source.is_dir():
        raise FileNotFoundError(f"No code snapshot in {args.run_dir}")
    shutil.make_archive(str(out / "training_source"), "gztar", source)
    metrics = [json.loads(line) for line in (args.run_dir / "epoch_metrics.jsonl").read_text().splitlines() if line]
    selected = next(row for row in metrics if row["epoch"] == int(state["epoch"]))
    dump(out / "gameplay_operating_points.json", {
        "schema_version": 1, "model_id": args.model_id,
        "score_semantics": "softmax probability of gameplay class index 1",
        "classifier": {"threshold": .5, "validation_split": "clean_val48", "f1": selected["f1"]},
        "selected_full_video": policy,
        "selection_caveat": "Retrospective full10 policy selection; outside-event footage is a proxy, not human non-gameplay labels."})
    (out / "evidence").mkdir()
    evidence = {"event_coverage.json": args.gate_root / "coverage/metrics.json",
                "pipeline_fp_impact.json": args.gate_root / "impact/pipeline_fp_impact.json",
                "validation_gate_result.json": args.gate_root / "impact/validation_gate_result.json",
                "val48_contract.json": REPO / "validation/gameplay_indoor_v8_val48_clean_v2.json"}
    for item in args.evidence:
        name, _, path = item.partition("=")
        evidence[name] = Path(path)
    for name, path in evidence.items():
        shutil.copy2(path, out / "evidence" / name)
    artifact = json.loads((args.template_bundle / "artifact.json").read_text())
    artifact.update(model_id=args.model_id, production=False, status="candidate", operating_points=policy)
    artifact["source"] = {"checkpoint_epoch": int(state["epoch"]), "checkpoint_sha256": sha256(checkpoint),
                          "training_run": args.run_dir.name, "dataset": args.dataset,
                          "initialization": "pretrained Qwen vision; no gameplay initializer"}
    artifact["validation"] = {
        "clean_val48_f1": selected["f1"], "clean_val48_windows": selected["count"],
        "confirmed_events": gate["confirmed_events"], "confirmed_events_excluded": gate["confirmed_events_excluded"],
        "rallies_fully_retained": gate["rallies_fully_retained"], "target_fp_current": gate["target_fp_current"],
        "target_fp_rejected": gate["target_fp_rejected"],
        "outside_event_footage_percent_total_match": gate["outside_event_footage_percent_total_match"],
        "full10_gate_status": gate["gate_status"],
        "label_caveat": "Model-assisted labels pending human acceptance; F1 provisional."}
    dump(out / "artifact.json", artifact)
    rehash(out)
    return {"bundle": str(out), "checkpoint_sha256": artifact["source"]["checkpoint_sha256"],
            "weights_sha256": sha256(out / "model.safetensors"), "artifact_sha256": sha256(out / "artifact.json")}


def parity(bundle: Path, scores_dir: Path, seconds: float, threshold: float, padding: float, gap: float) -> dict:
    """Score the first seconds of one full10 match with the pipeline runtime and compare (smoke.py)."""
    import csv
    import numpy as np
    from skyball_pipeline.gameplay_qwen import score_qwen_gameplay_video
    from skyball_pipeline.gameplay_trt import validate_gameplay_bundle

    validate_gameplay_bundle(bundle)
    work = bundle.parent / f"{bundle.name}_parity"
    work.mkdir(exist_ok=False)
    meta = json.loads((scores_dir / "metadata.json").read_text())
    score_qwen_gameplay_video(
        video=Path(meta["video"]), engine_path=bundle / "model.safetensors", artifact_json=bundle / "artifact.json",
        gameplay_csv=work / "gameplay.csv", scores_csv=work / "scores.csv", timing_json=work / "timing.json",
        max_frames=round(meta["video_info"]["fps"] * seconds), threshold=threshold, merge_gap_sec=gap,
        padding_sec=padding, sample_fps=1, batch_size=4, min_start_frame=0)
    ours = list(csv.DictReader((work / "scores.csv").open()))
    ref = list(csv.DictReader((scores_dir / "scores.csv").open()))[:len(ours)]

    def column(rows):
        return np.array([float(r.get("score_gameplay", r.get("gameplay_score", r.get("score")))) for r in rows])

    x, y = column(ours), column(ref)
    result = {"windows": len(ours), "max_abs_score_difference": float(abs(x - y).max()),
              "threshold_decision_differences": int(((x >= threshold) != (y >= threshold)).sum())}
    (work / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["max_abs_score_difference"] >= .005 or result["threshold_decision_differences"]:
        raise ValueError(f"Runtime parity failed: {result}")
    shutil.copy2(work / "parity.json", bundle / "evidence/runtime_parity.json")
    rehash(bundle)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="default: <run-dir>/best_*.pth.tar")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--dataset", default="v8_val48_trainclean_overlay_20261006_r4")
    parser.add_argument("--template-bundle", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new bundle directory")
    parser.add_argument("--threshold", type=float, default=.9)
    parser.add_argument("--padding-sec", type=float, default=5.)
    parser.add_argument("--merge-gap-sec", type=float, default=0.)
    parser.add_argument("--evidence", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--allow-failed-gate", action="store_true")
    parser.add_argument("--parity-scores", type=Path,
                        help="full10_scores/<match> dir of this checkpoint; runs the pipeline-runtime parity check")
    parser.add_argument("--parity-seconds", type=float, default=120.)
    args = parser.parse_args()
    receipt = export(args)
    if args.parity_scores:
        receipt["parity"] = parity(args.output, args.parity_scores, args.parity_seconds,
                                   args.threshold, args.padding_sec, args.merge_gap_sec)
        receipt["artifact_sha256"] = sha256(args.output / "artifact.json")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    sys.exit(main())
