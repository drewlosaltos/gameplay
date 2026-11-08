# Qwen V8 production run, 2026-10-06/07: one-off scripts as they ran

These scripts trained, validated and published
`v8_val48_qwen672x384_single_stage_fp8_20261006_r3`. They are copied unchanged
from `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/` and
`/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1/`. They are a
record, not tools. Their paths are hardcoded, several of those paths have since
moved, and most of them wait on status files from earlier stages or refuse to run
once their output exists. To reproduce the model, follow `REPRODUCE.md` at the
repo root, which uses the reusable versions in `scripts/`.

| Script | Stage | Reusable version |
|---|---|---|
| `launch_single_stage_v8_when_ready.py` | Waited for the pilots, copied `skyball_gameplay/` into `<run>/code_snapshot`, wrote `<run>/config.json` and ran `python -m skyball_gameplay.training` in `skyball-prod-infer` with the TorchAO overlay on `PYTHONPATH` | `scripts/train_qwen_canonical.py` + `config/indoor_canonical.json` (same arguments) |
| `validate_single_stage_when_ready.py` | Classification on val50 (cohorts), full10 scoring on shards 0-1, score view, and the gate at the common 0.5/20 s policy | `skyball_gameplay.evaluation`, `scripts/score_gameplay_pipeline_inputs.py`, `scripts/make_gate_score_view.py`, gate scripts |
| `run_single_stage_full10_comparison.py` | Calls the functions above and writes `full10_common_policy_vs_prod.json` (candidate vs previous production at 0.5/20 s) | Same as above |
| `search_single_stage_policy_when_ready.py` | Searched 272 policies (17 thresholds x 8 paddings x 2 gaps) and selected 0.9 / 5 s / gap 0; wrote `selected_policy/` | The fixed-policy commands in `REPRODUCE.md` |
| `export.py` | Converted the epoch-3 checkpoint to the FP32 `model.safetensors` bundle, using the Qwen v5 bundle as the template for `artifact.json` and `qwen_vision_config.json` | `scripts/export_qwen_gameplay_bundle.py` |
| `smoke.py` | 120 s pipeline-runtime parity on `eval_club_005_pac_balboa` (117 windows, max diff 0.000277) | `scripts/export_qwen_gameplay_bundle.py --parity-scores` |
| `finalize_bundle.py` | Added the parity and val48 contract evidence and re-hashed `artifact.json` | Folded into `scripts/export_qwen_gameplay_bundle.py` |

The val48 classification step in `validate_single_stage_when_ready.py` was
terminated. The reported clean val48 metrics are the in-training FP32
validation in `<run>/epoch_metrics.jsonl`, collected in
`v8_workflow_20261006/single_stage_clean_val48_vs_prod.json`.
