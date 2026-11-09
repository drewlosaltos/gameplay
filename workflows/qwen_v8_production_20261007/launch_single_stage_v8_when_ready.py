#!/usr/bin/env python3
"""Conditionally train a full-v8 Qwen model from the pretrained encoder."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val48_trainclean_overlay_20261006_r4")
REPO = Path("/home/atao/vsdevel/skyball/gameplay")
RUN = Path("/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_single_stage_4gpu_20261006_r3")
QWEN = Path("/home/atao/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989")
OVERLAY = Path("/mnt/t/output/skyball_gameplay/v8_qwen_fp8_prototype_20261006_r1/site")
TAG = "qwen672x384_v8_single_stage_4gpu"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def status(stage: str, **details: object) -> None:
    write(WORK / "single_stage_v8_status.json", {
        "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **details,
    })


def pilot_decision(pilot: dict) -> dict:
    if pilot.get("status") != "complete":
        raise RuntimeError("Four-epoch single-stage pilot is incomplete")
    series = [float(value) for value in pilot["f1_by_epoch"]]
    if len(series) != 4 or not all(math.isfinite(value) for value in series):
        raise RuntimeError("Four-epoch pilot has invalid F1 metrics")
    reference_native = float(pilot["reference_val8_f1"]["legacy_initializer"])
    highres = pilot["reference_val8_f1"].get("legacy_initializer_highres")
    reference_highres = float(highres) if highres is not None else None
    reference = max(reference_native, reference_highres) if reference_highres is not None else reference_native
    best = max(series)
    threshold = max(.8, reference - .06)
    return {
        "train_full_v8": best >= threshold,
        "rule": "Pilot best F1 >= max(0.8, best legacy-initializer F1 at native/high resolution minus 0.06) on the same fixed val8 split",
        "pilot_best_f1": best, "pilot_last_epoch_f1": series[-1],
        "legacy_initializer_f1": reference, "legacy_initializer_native_f1": reference_native,
        "legacy_initializer_highres_f1": reference_highres, "required_f1": threshold,
        "encoder_lr": pilot["encoder_lr"], "acceleration": pilot["acceleration"],
        "pilot_evidence": str(WORK / "single_stage_four_epoch_result.json"),
    }


def batch_choice() -> dict | None:
    batch20_path = WORK / "batch20_four_epoch_result.json"
    if batch20_path.exists():
        batch20 = json.loads(batch20_path.read_text())
        if batch20.get("status") != "complete":
            raise RuntimeError(f"Batch-20 four-epoch result is incomplete: {batch20_path}")
        choice20 = batch20.get("decision", {}).get("use_for_full_v8_challenger")
        if not isinstance(choice20, bool):
            raise RuntimeError(f"Batch-20 pilot lacks its decision: {batch20_path}")
        if choice20:
            return {"global_microbatch": 20,
                    "reason": batch20["decision"]["rule"],
                    "evidence": str(batch20_path),
                    "batch20_four_epoch_best_f1": batch20["best_f1"],
                    "batch16_four_epoch_best_f1": batch20["batch16_best_f1"],
                    "batch20_steady_throughput_gain_pct":
                    batch20["steady_epoch2_gain_vs_batch16_pct"]}
    result_path = WORK / "batch16_four_epoch_result.json"
    status_path = WORK / "batch16_four_epoch_status.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"Batch-16 four-epoch result is incomplete: {result_path}")
        decision = result.get("decision", {})
        use_batch16 = decision.get("use_for_full_v8_challenger")
        if not isinstance(use_batch16, bool):
            raise RuntimeError(f"Batch-16 four-epoch result lacks its decision: {result_path}")
        return {"global_microbatch": 16 if use_batch16 else 8,
                "reason": decision["rule"], "evidence": str(result_path),
                "batch16_four_epoch_best_f1": result["best_f1"],
                "batch8_four_epoch_best_f1": result["batch8_four_epoch_best_f1"],
                "batch16_steady_throughput_gain_pct":
                    result["batch16_vs_batch8_steady_epoch2_gain_pct"]}
    if status_path.exists():
        state = json.loads(status_path.read_text())
        if state.get("stage") in {"deferred", "skipped_upstream_incomplete", "failed"}:
            return {"global_microbatch": 8,
                    "reason": f"Optional batch-16 pilot {state['stage']}; retain validated batch-8 recipe",
                    "evidence": str(status_path)}
        if state.get("stage") == "complete":
            raise RuntimeError(f"Batch-16 pilot reports complete without its result: {result_path}")
    return None


def compile_choice() -> dict:
    result_path = WORK / "reduce_overhead_fp8_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"Reduce-overhead result is incomplete: {result_path}")
        selected = result.get("decision", {}).get("use_for_full_v8_challenger")
        if not isinstance(selected, bool):
            raise RuntimeError(f"Reduce-overhead pilot lacks a decision: {result_path}")
        return {"mode": "reduce-overhead" if selected else "default",
                "reason": result["decision"]["rule"], "evidence": str(result_path),
                "best_f1": result["best_f1"],
                "steady_epoch2_gain_pct": result["steady_epoch2_gain_vs_default_pct"]}
    return {"mode": "default", "reason": "No qualifying reduce-overhead pilot result",
            "evidence": str(WORK / "reduce_overhead_fp8_status.json")}


def wait_for_turn() -> tuple[dict, dict, dict]:
    while True:
        failures = {
            "stage2_status.json": "Stage-two training",
            "stage2_validation_status.json": "Stage-two validation",
            "single_stage_four_epoch_status.json": "Single-stage pilot",
            "pilot_reference_highres_status.json": "High-resolution pilot reference",
        }
        for name, label in failures.items():
            path = WORK / name
            if path.exists() and json.loads(path.read_text()).get("stage") == "failed":
                raise RuntimeError(f"{label} failed; inspect {path}")
        ready = WORK / "val48_trainclean_dataset_ready.json"
        pilot = WORK / "single_stage_four_epoch_result.json"
        stage2 = WORK / "stage2_validation_status.json"
        reference = WORK / "pilot_reference_highres_status.json"
        if all(path.exists() for path in (ready, pilot, stage2, reference)):
            training_status = json.loads(stage2.read_text()).get("stage")
            reference_status = json.loads(reference.read_text()).get("stage")
            if (training_status == "common_policy_complete"
                    and reference_status == "complete"):
                dataset = json.loads(ready.read_text())
                if dataset.get("status") != "ready" or dataset.get("dataset") != str(DATA):
                    raise RuntimeError("Dataset ready marker disagrees with expected v8 dataset")
                choice = batch_choice()
                if choice is not None:
                    return dataset, json.loads(pilot.read_text()), choice
        time.sleep(30)


def main() -> None:
    status("waiting_for_pilot_dataset_and_stage2_common_gate")
    dataset, pilot, batch = wait_for_turn()
    decision = pilot_decision(pilot)
    decision["batch_choice"] = batch
    compile_mode = compile_choice()
    decision["compile_choice"] = compile_mode
    write(WORK / "single_stage_v8_decision.json", decision)
    if not decision["train_full_v8"]:
        status("deferred_after_pilot", decision=str(WORK / "single_stage_v8_decision.json"))
        return
    if RUN.exists():
        raise FileExistsError(f"Versioned run directory already exists: {RUN}")
    if not QWEN.is_dir() or not (OVERLAY / "torchao").is_dir():
        raise RuntimeError("Pretrained Qwen snapshot or TorchAO overlay is missing")
    if dataset["window_count"] != json.loads((DATA / "metadata/summary.json").read_text())["window_count"]:
        raise RuntimeError("Dataset ready count differs from metadata")
    lr_multiplier = {"1e-5": ".1", "3e-6": ".03", "1e-6": ".01"}[decision["encoder_lr"]]
    gpu_before = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader",
    ], capture_output=True, text=True, check=True).stdout
    RUN.mkdir(parents=True)
    snapshot = RUN / "code_snapshot"
    snapshot.mkdir()
    shutil.copytree(REPO / "skyball_gameplay", snapshot / "skyball_gameplay",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args = [
        str(DATA), "--arch", "qwen_vision", "--qwen-snapshot", str(QWEN),
        "--optimizer", "adamw", "--lr", "1e-4", "--backbone-lr-multiplier", lr_multiplier,
        "--weight-decay", "1e-4", "--epochs", "4", "--lr-scheduler", "step",
        "--lr-step-size", "2", "--lr-gamma", ".1", "--warmup-epochs", ".5",
        "--max-grad-norm", "1", "--amp", "--amp-dtype", "bfloat16", "--validation-fp32",
        "--training-acceleration", decision["acceleration"], "--compile-mode", compile_mode["mode"],
        "--dist-url", f"tcp://127.0.0.1:{port}", "--dist-backend", "nccl",
        "--multiprocessing-distributed", "--world-size", "1", "--rank", "0",
        "-j", "16", "-b", str(batch["global_microbatch"]),
        "--grad-accum-steps", "1", "--input-frames", "4",
        "--label-mode", "binary", "--image-size", "384", "672",
        "--resize-mode", "letterbox", "--train-geometry", "full_frame",
        "--aug-policy", "none", "--random-erasing", "0", "--color-jitter", ".1",
        "--temporal-dropout", ".1", "--temporal-shared-augment", "--rotation-degrees", "5",
        "--selection-metric", "f1", "--seed", "20260916", "--print-freq", "100",
        "--run-tag", TAG, "--output-dir", str(RUN),
    ]
    write(RUN / "config.json", {
        "args": args, "dataset": str(DATA),
        "dataset_windows_sha256": sha256(DATA / "metadata/windows.jsonl"),
        "dataset_summary_sha256": sha256(DATA / "metadata/summary.json"),
        "qwen_snapshot": str(QWEN), "initializer": "pretrained Qwen vision encoder; no local gameplay checkpoint",
        "code_snapshot": str(snapshot),
        "code_files_sha256": {str(path.relative_to(snapshot)): sha256(path) for path in sorted(snapshot.rglob("*.py"))},
        "pilot_decision": str(WORK / "single_stage_v8_decision.json"),
        "gpu_before_launch": gpu_before, "gpus": "0,1,2,3",
        "global_microbatch": batch["global_microbatch"], "grad_accum_steps": 1,
        "effective_batch": batch["global_microbatch"], "batch_choice": batch,
        "compile_choice": compile_mode,
        "selection": "clean v8 val48 F1; full val20 and val50 scored separately for historical continuity",
    })
    env = os.environ | {
        "CUDA_VISIBLE_DEVICES": "0,1,2,3", "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{OVERLAY}:{snapshot}",
    }
    status("training", run=str(RUN), acceleration=decision["acceleration"],
           compile_mode=compile_mode["mode"],
           encoder_lr=decision["encoder_lr"], global_microbatch=batch["global_microbatch"])
    start = time.monotonic()
    with (RUN / "launcher.log").open("x") as log:
        result = subprocess.run([
            "conda", "run", "--no-capture-output", "-n", "skyball-prod-infer",
            "python", "-m", "skyball_gameplay.training", *args,
        ], cwd=snapshot, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Single-stage training failed with exit {result.returncode}: {RUN / 'launcher.log'}")
    metrics = [json.loads(line) for line in (RUN / "epoch_metrics.jsonl").read_text().splitlines() if line]
    if len(metrics) != 4 or not all(math.isfinite(float(row["f1"])) for row in metrics):
        raise RuntimeError("Four valid single-stage epoch metrics were not saved")
    best = RUN / f"best_{TAG}.pth.tar"
    if not best.is_file():
        raise RuntimeError("Single-stage training finished without a best checkpoint")
    status("training_complete", run=str(RUN), checkpoint=str(best),
           checkpoint_sha256=sha256(best), best_epoch=max(metrics, key=lambda row: row["f1"])["epoch"],
           best_clean_val48_f1=max(row["f1"] for row in metrics), elapsed_sec=time.monotonic() - start)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
