#!/usr/bin/env python3
"""Search safe full10 temporal policies for the v8 Qwen single-stage candidate."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/atao/vsdevel/skyball/gameplay")
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
from evaluate_gameplay_gate import Policy, load_match_data
from evaluate_gameplay_pipeline_fp_impact import analyze_match, aggregate_impact, write_csv, write_markdown
from validate_gameplay_gate import evaluate_contract, markdown_result

WORK = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
ROOT = Path("/mnt/t/output/skyball_gameplay/v8_single_stage_val48_trainclean_validation_20261007_r3")
PRIMARY = Path("/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1")
FP_SUITE = Path("/mnt/t/output/skyball_pipeline/evaluation_suite_serve_primary_e2e_cached_detector_v1_20260717")
VIEW = ROOT / "gate_score_view"
OUT = ROOT / "selected_policy"
PROD_SCORE = Path("/mnt/t/output/skyball_gameplay/human_app_v5_qwen672x384_2gpu_20260925_r1/full10_scores")
SUITE = Path("/mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1")


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def status(stage: str, **details: object) -> None:
    write(WORK / "single_stage_policy_status.json", {"stage": stage,
          "updated_at": datetime.now(timezone.utc).isoformat(), **details})


def safe(aggregate: dict) -> bool:
    game = aggregate["gameplay"]
    return (game["annotated_gameplay_recall"] >= 1 - 1e-9
            and game["event_recall"] >= 1 - 1e-9
            and game["rally_full_coverage_recall"] >= 1 - 1e-9
            and aggregate["matched_true_actions"]["rejected"] == 0
            and aggregate["valid_predicted_plays"]["rejected"] == 0)


def main() -> None:
    status("waiting_for_common_policy")
    while True:
        if ((ROOT / "common_policy_summary.json").exists()
                or (ROOT / "full10_common_policy_vs_prod.json").exists()):
            break
        validation = WORK / "single_stage_validation_status.json"
        if validation.exists():
            stage = json.loads(validation.read_text())["stage"]
            if stage == "failed":
                raise RuntimeError("Common-policy validation failed")
            if stage == "skipped_after_pilot":
                status("skipped_after_pilot")
                return
        time.sleep(30)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    status("searching", grid="17 thresholds x 8 paddings x 2 gaps")
    matches = load_match_data(PRIMARY, VIEW, 3.0)
    suite = json.loads((FP_SUITE / "evaluation_suite_summary.json").read_text())
    suite_summary = {row["match"]: row["summary"] for row in suite["matches"]}
    contract = json.loads((REPO / "validation/gameplay_gate_eval10_v2.json").read_text())
    thresholds = [.05, .075, .1, .125, .15, .175, .2, .225, .25, .3, .35, .4, .5,
                  .6, .7, .8, .9]
    paddings = [5., 10., 15., 20., 30., 45., 60., 90.]
    gaps = [0., 10.]
    grid = []
    best = None
    for threshold in thresholds:
        for padding in paddings:
            for gap in gaps:
                policy = Policy(threshold=threshold, merge_gap_sec=gap,
                                pad_before_sec=padding, pad_after_sec=padding)
                rows = [analyze_match(match, VIEW, FP_SUITE, suite_summary, policy) for match in matches]
                aggregate = aggregate_impact(rows)
                report = {"schema_version": 1,
                          "scope": "standalone retrospective quality-impact analysis",
                          "pipeline_modified": False, "pipeline_root": str(FP_SUITE),
                          "study_root": str(VIEW), "policy": asdict(policy),
                          "selection_caveat": "Policy selected on these ten matches; retrospective validation, not an independent test.",
                          "per_match": rows, "aggregate": aggregate}
                gate = evaluate_contract(report, contract)
                if gate["status"] == "invalid_population":
                    raise RuntimeError("Canonical ten-match population changed")
                eligible = safe(aggregate)
                target = aggregate["outside_gt_event_margin_action_false_positives"]
                grid.append({"policy": asdict(policy), "safe": eligible,
                             "gate_status": gate["status"], "target_fp_current": target["current"],
                             "target_fp_rejected": target["rejected"],
                             "target_fp_remaining": target["remaining"],
                             "events_recall": aggregate["gameplay"]["event_recall"],
                             "annotated_gameplay_recall": aggregate["gameplay"]["annotated_gameplay_recall"],
                             "rally_full_coverage_recall": aggregate["gameplay"]["rally_full_coverage_recall"],
                             "matched_actions_rejected": aggregate["matched_true_actions"]["rejected"],
                             "valid_plays_rejected": aggregate["valid_predicted_plays"]["rejected"]})
                if eligible:
                    rank = (target["rejected"], aggregate["invalid_predicted_plays"]["rejected"],
                            aggregate["gameplay"]["excluded_video_fraction"])
                    if best is None or rank > best[0]:
                        best = (rank, report, gate)
    write(OUT / "searched_grid.json", {
        "selection_rule": "All safety recalls=100% and no rejected matched actions/valid plays; then maximize target FP rejection",
        "contract": str(REPO / "validation/gameplay_gate_eval10_v2.json"), "policies": grid})
    if best is None:
        result = {"status": "no_safe_policy", "searched": len(grid),
                  "evidence": str(OUT / "searched_grid.json")}
        write(OUT / "result.json", result)
        status("no_safe_policy", result=str(OUT / "result.json"))
        return
    _, report, gate = best
    write(OUT / "pipeline_fp_impact.json", report)
    write_csv(OUT / "pipeline_fp_impact.csv", report["per_match"])
    write_markdown(OUT / "pipeline_fp_impact.md", report)
    write(OUT / "validation_gate_result.json", gate)
    (OUT / "validation_gate_result.md").write_text(markdown_result(gate))
    status("selected_coverage", policy=report["policy"])
    coverage = {}
    for label, score_root in (("candidate", ROOT / "full10_scores"),
                              ("current_production", PROD_SCORE)):
        destination = OUT / f"coverage_{label}"
        command = [sys.executable, str(REPO / "scripts/evaluate_gameplay_event_coverage.py"),
                   "--pipeline-root", str(SUITE), "--candidate-scores", str(score_root),
                   "--output-root", str(destination),
                   "--candidate-threshold", str(report["policy"]["threshold"]),
                   "--candidate-padding-sec", str(report["policy"]["pad_before_sec"]),
                   "--candidate-merge-gap-sec", str(report["policy"]["merge_gap_sec"])]
        with (OUT / f"coverage_{label}.log").open("x") as log:
            outcome = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        if outcome.returncode:
            raise RuntimeError(f"Selected-policy coverage failed for {label}: {outcome.returncode}")
        metrics = json.loads((destination / "metrics.json").read_text())
        row = metrics["aggregate"]["candidate_selected_policy"]
        coverage[label] = {"events": row["events"], "events_excluded": row["events_excluded"],
                           "outside_event_margin_admitted_pct_match": row["outside_event_margin_admitted_pct_match"],
                           "evidence": str(destination / "metrics.json")}
    target = report["aggregate"]["outside_gt_event_margin_action_false_positives"]
    result = {"status": "complete", "searched": len(grid),
              "safe_policies": sum(row["safe"] for row in grid),
              "selected_policy": report["policy"], "gate_status": gate["status"],
              "target_fp_rejected": target["rejected"], "target_fp_current": target["current"],
              "target_fp_remaining": target["remaining"], "coverage": coverage,
              "evidence": str(OUT)}
    write(OUT / "result.json", result)
    status("complete", result=str(OUT / "result.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
