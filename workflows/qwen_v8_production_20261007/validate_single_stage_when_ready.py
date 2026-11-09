#!/usr/bin/env python3
"""Evaluate the v8 single-stage candidate and production on identical inputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
ROOT = Path("/mnt/t/output/skyball_gameplay/v8_single_stage_val48_trainclean_validation_20261007_r3")
STAGE2 = Path("/mnt/t/output/skyball_gameplay/v8_stage2_val48_trainclean_validation_20261006_r3")
REPO = Path("/home/atao/vsdevel/skyball/gameplay")
PIPE = Path("/home/atao/vsdevel/skyball/pipeline")
DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val48_trainclean_overlay_20261006_r4")
EVAL_DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")
LEGACY = Path("/mnt/t/data/vball/skyball/gameplay/v5")
RUN = Path("/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_single_stage_4gpu_20261006_r3")
TAG = "qwen672x384_v8_single_stage_4gpu"
CANDIDATE = RUN / f"best_{TAG}.pth.tar"
PROD = Path("/mnt/t/output/trn/skyball_gameplay/human_app_v5_val20_qwen672x384_rotate5_e4_2gpu_b4a2_20260925_r1/best_qwen672x384_v5_e4_2gpu_b4a2.pth.tar")
SUITE = Path("/mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1")
PRIMARY = Path("/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1")
FP_SUITE = Path("/mnt/t/output/skyball_pipeline/evaluation_suite_serve_primary_e2e_cached_detector_v1_20260717")
PROD_SCORE = Path("/mnt/t/output/skyball_gameplay/human_app_v5_qwen672x384_2gpu_20260925_r1/full10_scores")
PROD_IMPACT = WORK / "prod_current_impact_050_20/pipeline_fp_impact.json"
PROD_COVER = WORK / "prod_current_coverage_050_20/metrics.json"


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def status(stage: str, **details: object) -> None:
    write(WORK / "single_stage_validation_status.json", {"stage": stage,
          "updated_at": datetime.now(timezone.utc).isoformat(), **details})


def command(args: list[str], log: Path, gpu: int | None = None, allowed=(0,)) -> int:
    env = os.environ | {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                        "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REPO}:{PIPE}"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with log.open("x") as out:
        result = subprocess.run(["conda", "run", "--no-capture-output", "-n", "skyball-prod-infer", "python", *args],
                                cwd=REPO, env=env, stdout=out, stderr=subprocess.STDOUT)
    if result.returncode not in allowed:
        raise RuntimeError(f"Command exited {result.returncode}; inspect {log}")
    return result.returncode


def wait_for_training() -> dict | None:
    while True:
        path = WORK / "single_stage_v8_status.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["stage"] == "training_complete":
                return state
            if state["stage"] == "deferred_after_pilot":
                return None
            if state["stage"] == "failed":
                raise RuntimeError("Single-stage training failed: " + state.get("error", ""))
        time.sleep(30)


def classification() -> dict:
    status("classification", training_dataset=str(DATA), evaluation_dataset=str(EVAL_DATA),
           production_reused_from=str(STAGE2))
    stage2 = json.loads((STAGE2 / "common_policy_summary.json").read_text())
    stage2_config = json.loads((Path(stage2["candidate_checkpoint"]).parent / "config.json").read_text())
    if (stage2.get("status") != "common_policy_complete"
            or stage2["production_sha256"] != sha256(PROD)
            or stage2["dataset"] != str(DATA)
            or stage2["classification_dataset"] != str(EVAL_DATA)
            or stage2["classification_manifest_sha256"] != sha256(EVAL_DATA / "metadata/windows.jsonl")
            or stage2_config["dataset_windows_sha256"] != sha256(DATA / "metadata/windows.jsonl")):
        raise RuntimeError("Stage-two production classification is not reusable on identical inputs")
    source_cohorts = STAGE2 / "classification/production/cohorts.json"
    source_legacy = STAGE2 / "classification/production_legacy_v5/metrics.json"
    source_predictions = STAGE2 / "classification/production/predictions.csv"
    production = stage2["classification"]["production"]
    if (not all(path.is_file() for path in (source_cohorts, source_legacy, source_predictions))
            or any(production[cohort] != json.loads(source_cohorts.read_text())[cohort]
                   for cohort in ("val20", "new30", "val50"))
            or production["legacy_v5"] != json.loads(source_legacy.read_text())["binary"]):
        raise RuntimeError("Stage-two production classification artifacts differ from its summary")
    write(ROOT / "production_classification_reuse.json", {
        "status": "pass", "checkpoint": str(PROD),
        "checkpoint_sha256": stage2["production_sha256"],
        "training_dataset": str(DATA), "training_windows_sha256": stage2_config["dataset_windows_sha256"],
        "evaluation_dataset": str(EVAL_DATA), "evaluation_windows_sha256": stage2["classification_manifest_sha256"],
        "cohorts": str(source_cohorts), "cohorts_sha256": sha256(source_cohorts),
        "legacy": str(source_legacy), "legacy_sha256": sha256(source_legacy),
        "predictions": str(source_predictions), "predictions_sha256": sha256(source_predictions),
    })
    jobs = []
    for gpu, (label, checkpoint, dataset, suffix) in enumerate((
        ("single_stage_v8", CANDIDATE, EVAL_DATA, ""),
        ("single_stage_v8", CANDIDATE, LEGACY, "_legacy_v5"),
    )):
        dest = ROOT / "classification" / f"{label}{suffix}"
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
                            "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
                            "PYTHONPATH": f"{REPO}:{PIPE}"}
        args = ["conda", "run", "--no-capture-output", "-n", "skyball-prod-infer", "python",
                "-m", "skyball_gameplay.evaluation", "--data", str(dataset),
                "--checkpoint", str(checkpoint), "--output-dir", str(dest),
                "-b", "64", "-j", "8"]
        if not suffix:
            args.append("--save-predictions")
        if (dest / "metrics.json").is_file():
            continue
        log = (ROOT / f"classification_{label}{suffix}.log").open("a")
        jobs.append((label, suffix, dest, subprocess.Popen(args, cwd=REPO, env=env,
                                                       stdout=log, stderr=subprocess.STDOUT), log))
    failures = []
    for label, suffix, dest, process, log in jobs:
        code = process.wait()
        log.close()
        if code or not (dest / "metrics.json").exists():
            failures.append((label, suffix, code))
    if failures:
        raise RuntimeError(f"Classification evaluation failures: {failures}; inspect {ROOT}/classification_*.log")
    outputs = {"production": production}
    for label in ("single_stage_v8",):
        dest = ROOT / "classification" / label
        if not (dest / "metrics.json").exists():
            raise RuntimeError(f"Classification metrics missing for {label}")
        command([str(REPO / "scripts/summarize_indoor_v8_cohorts.py"),
                 "--predictions", str(dest / "predictions.csv"),
                 "--exclude-validation-sources", str(WORK / "v8_validation_leak_audit.json"),
                 "--output", str(dest / "cohorts.json")], ROOT / f"cohorts_{label}.log")
        outputs[label] = json.loads((dest / "cohorts.json").read_text())
        legacy = ROOT / "classification" / f"{label}_legacy_v5"
        outputs[label]["legacy_v5"] = json.loads((legacy / "metrics.json").read_text())["binary"]
    return outputs


def score_full10(checkpoint_sha: str) -> list[dict]:
    status("full10_scoring", checkpoint_sha256=checkpoint_sha)
    score_root = ROOT / "full10_scores"
    score_root.mkdir(exist_ok=True)
    jobs = []
    # Shards 2 and 3 can be scored concurrently while held-out classification uses GPUs 0 and 1.
    # Finish shards 0 and 1 here, then validate all ten score artifacts below.
    for gpu in range(2):
        env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
                            "OPENBLAS_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
                            "PYTHONPATH": f"{REPO}:{PIPE}"}
        log = (ROOT / f"score_shard_{gpu}.log").open("x")
        args = ["conda", "run", "--no-capture-output", "-n", "skyball-prod-infer", "python",
                str(REPO / "scripts/score_gameplay_pipeline_inputs.py"),
                "--shard", str(gpu), "--num-shards", "4", "--checkpoint", str(CANDIDATE),
                "--pipeline-root", str(SUITE), "--output-root", str(score_root)]
        jobs.append((gpu, subprocess.Popen(args, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    failures = []
    for gpu, process, log in jobs:
        code = process.wait()
        log.close()
        if code:
            failures.append((gpu, code))
    if failures:
        raise RuntimeError(f"Full10 scoring shard failures {failures}; see {ROOT}/score_shard_*.log")
    rows = []
    for manifest_path in sorted(SUITE.glob("*/manifest.json")):
        meta = json.loads((score_root / manifest_path.parent.name / "metadata.json").read_text())
        manifest = json.loads(manifest_path.read_text())
        if meta["checkpoint_sha256"] != checkpoint_sha:
            raise RuntimeError("Mixed checkpoint full10 scores")
        if Path(meta["video"]).resolve() != Path(manifest["video"]).resolve():
            raise RuntimeError("Full10 score video differs from canonical input")
        if meta["gt_sha256"] != manifest["inputs"]["gt_sync_csv"]["sha256"]:
            raise RuntimeError("Full10 score GT differs from canonical input")
        if meta["score_count"] != json.loads((manifest_path.parent / "gameplay_timing.json").read_text())["windows"]:
            raise RuntimeError("Full10 score count differs")
        rows.append(meta)
    if len(rows) != 10:
        raise RuntimeError(f"Expected ten full10 matches; got {len(rows)}")
    return rows


def make_score_view(rows: list[dict]) -> Path:
    view = ROOT / "gate_score_view"
    for meta in rows:
        match = Path(meta["source_manifest"]).parent.name
        folder = view / match
        folder.mkdir(parents=True)
        source = ROOT / "full10_scores" / match / "scores.csv"
        (folder / "scores.csv").symlink_to(source)
        write(folder / "score_metadata.json", {
            "timing": {"duration_sec": meta["video_info"]["frame_count"] / meta["video_info"]["fps"]},
            "window_sec": 4, "exact_input_metadata": str(ROOT / "full10_scores" / match / "metadata.json"),
            "checkpoint_sha256": meta["checkpoint_sha256"], "scores_sha256": sha256(source),
        })
    return view


def gate(checkpoint_sha: str, view: Path) -> dict:
    status("full10_gate", checkpoint_sha256=checkpoint_sha)
    coverage = ROOT / "candidate_coverage_common_050_20"
    command([str(REPO / "scripts/evaluate_gameplay_event_coverage.py"),
             "--pipeline-root", str(SUITE), "--candidate-scores", str(ROOT / "full10_scores"),
             "--output-root", str(coverage), "--candidate-threshold", ".5",
             "--candidate-padding-sec", "20"], ROOT / "candidate_coverage_common_050_20.log")
    impact = ROOT / "candidate_impact_common_050_20"
    command([str(REPO / "scripts/evaluate_gameplay_pipeline_fp_impact.py"),
             "--evaluation-root", str(PRIMARY), "--study-root", str(view),
             "--pipeline-root", str(FP_SUITE), "--output-dir", str(impact),
             "--threshold", ".5", "--merge-gap-sec", "0", "--padding-sec", "20"],
            ROOT / "candidate_impact_common_050_20.log")
    command([str(REPO / "scripts/validate_gameplay_gate.py"),
             "--report", str(impact / "pipeline_fp_impact.json"),
             "--contract", str(REPO / "validation/gameplay_gate_eval10_v2.json"),
             "--output", str(impact / "validation_gate_result.json")],
            ROOT / "candidate_gate_common_050_20.log", allowed=(0, 1, 2))
    for path in (PROD_COVER, PROD_IMPACT):
        if not path.exists():
            raise RuntimeError(f"Current-production baseline report missing: {path}")
    return {
        "production_coverage": json.loads(PROD_COVER.read_text()),
        "candidate_coverage": json.loads((coverage / "metrics.json").read_text()),
        "production_impact": json.loads(PROD_IMPACT.read_text()),
        "candidate_impact": json.loads((impact / "pipeline_fp_impact.json").read_text()),
        "candidate_contract": json.loads((impact / "validation_gate_result.json").read_text()),
    }


def main() -> None:
    training = json.loads((WORK / "single_stage_v8_status.json").read_text())
    if training.get("stage") != "training_complete":
        status("waiting_for_single_stage_training", run=str(RUN))
        training = wait_for_training()
    if training is None:
        status("skipped_after_pilot", decision=str(WORK / "single_stage_v8_decision.json"))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if training["checkpoint_sha256"] != sha256(CANDIDATE):
        raise RuntimeError("Trained checkpoint checksum differs from launch status")
    config = json.loads((RUN / "config.json").read_text())
    if config["dataset_windows_sha256"] != sha256(DATA / "metadata/windows.jsonl"):
        raise RuntimeError("Dataset manifest changed since training")
    prod_cache = json.loads((WORK / "prod_full10_cache_verified.json").read_text())
    if prod_cache["checkpoint_sha256"] != sha256(PROD) or prod_cache["canonical_primary_root"] != str(PRIMARY):
        raise RuntimeError("Production full10 cache identity changed")
    classes = classification()
    scores = score_full10(training["checkpoint_sha256"])
    view = make_score_view(scores)
    reports = gate(training["checkpoint_sha256"], view)
    cover = {label: reports[label]["aggregate"]["candidate_selected_policy"]
             for label in ("production_coverage", "candidate_coverage")}
    impact = {label: reports[label]["aggregate"] for label in ("production_impact", "candidate_impact")}
    result = {
        "status": "common_policy_complete", "candidate_checkpoint": str(CANDIDATE),
        "candidate_sha256": training["checkpoint_sha256"],
        "production_checkpoint": str(PROD), "production_sha256": prod_cache["checkpoint_sha256"],
        "dataset": str(DATA), "classification_dataset": str(EVAL_DATA),
        "classification_manifest_sha256": sha256(EVAL_DATA / "metadata/windows.jsonl"),
        "policy": {"threshold": .5, "padding_sec_per_side": 20, "merge_gap_sec": 0},
        "classification": classes, "coverage": cover,
        "gate": {label: {"annotated_gameplay_recall": value["gameplay"]["annotated_gameplay_recall"],
                         "event_recall": value["gameplay"]["event_recall"],
                         "rally_full_coverage_recall": value["gameplay"]["rally_full_coverage_recall"],
                         "target_fp_current": value["outside_gt_event_margin_action_false_positives"]["current"],
                         "target_fp_rejected": value["outside_gt_event_margin_action_false_positives"]["rejected"],
                         "target_fp_remaining": value["outside_gt_event_margin_action_false_positives"]["remaining"],
                         "matched_actions_rejected": value["matched_true_actions"]["rejected"],
                         "valid_plays_rejected": value["valid_predicted_plays"]["rejected"]}
                 for label, value in impact.items()},
        "candidate_contract": reports["candidate_contract"],
        "candidate_full10_runtime_sec_sum": sum(row["elapsed_sec"] for row in scores),
        "evidence": str(ROOT),
    }
    write(ROOT / "common_policy_summary.json", result)
    status("common_policy_complete", result=str(ROOT / "common_policy_summary.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
