# Qwen V8 single-stage production promotion — 2026-10-07

The user approved `v8_val48_qwen672x384_single_stage_fp8_20261006_r3` as the indoor gameplay default on pipeline/main,
and its single-stage recipe and clean val48 split as gameplay/main defaults.
This replaces the Qwen v5 checkpoint; beach defaults are unchanged.

Training run: `/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_single_stage_4gpu_20261006_r3`. Selected checkpoint: epoch 3 of 4,
SHA256 `67f342a391234c85b25b6b14a1a12d3034d5acd7590677b4b89d0c9d0be0600f`. Four-frame Qwen3.6 at 672x384 with temporal
mean pooling, trained fully unfrozen directly from pretrained Qwen vision.
Four GPUs, global batch 16, TorchAO tensorwise FP8/compiled training, FP32
validation; encoder LR 3e-6, head LR 1e-4. Exact launch settings are in
`config/indoor_canonical.json` in gameplay. No legacy adaptation stage.

| Metric | Previous Qwen v5 | Qwen V8 | Better |
|---|---:|---:|---|
| Clean val48 F1 (16,281 windows) | 94.6217% | 96.1224% | Higher |
| Selected gameplay threshold | 0.5 | 0.9 | Policy |
| Padding per side / merge gap | 20s / 0s | 5s / 0s | Policy |
| Confirmed events excluded (8,848) | 0 | 0 | Lower |
| Full rallies retained (1,370) | 100% | 100% | Higher |
| Matched true actions / valid plays rejected | 0 / 0 | 0 / 0 | Lower |
| Target false positives rejected (114) | 94 | 94 | Higher |
| Admitted footage | 713.2163 min | 659.7515 min | Lower, subject to safety |
| Outside-event footage / entire match | 45.1950% | 39.6748% | Lower, subject to safety |

Production was compared at its selected existing policy; only the candidate
policy was searched in this study. The same ten matches were used for policy
selection and assessment. Outside-event footage is a proxy, not manually labeled
non-gameplay. Clean val48 labels remain provisional pending human acceptance.
At common 0.5/20s settings the candidate rejected 92/114 target FPs versus
94/114 for production: the selected model and policy must travel together.

The published FP32 safetensors bundle is at
`googdrive:skyball/checkpoints/gameplay/v8_val48_qwen672x384_single_stage_fp8_20261006_r3`. Inference stays on the existing
PyTorch Qwen runtime with cached frame embeddings, sample rate 1 Hz and frame
batch 4. FP8 applies to training only. A 120-second pipeline-runtime comparison
covered 117 windows: maximum score difference 0.000276745, zero differences
at threshold 0.9. Weights SHA256 `fe1bf1893278ab1946b03038e1ed8c6225883b4c604237af268c93b63fbb5e9a`.

Evidence:
- Classification: `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/single_stage_clean_val48_vs_prod.json`
- Full10 scores, comparison and selected-policy gate: `/mnt/t/output/skyball_gameplay/v8_single_stage_val48_trainclean_validation_20261007_r3`
- Publication, upload verification and runtime parity: `/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1`

This promotion changes the gameplay asset and its policy only. Other assets
and policies match pipeline origin/main at `bd54dd7`. Full10 gameplay inference
and retrospective admission replay were completed; no fresh end-to-end
pipeline evaluation was run for this promotion.

Rollback: restore `human_app_v5_qwen672x384_rotate5_e4_20260925_r1` with threshold
0.5, 20 seconds of padding per side and zero merge gap. `--no-gameplay-gate`
remains an emergency bypass. Published artifacts preserve the previous model.

Clean val48 is the default for new training and model selection. No val20 or
val50 scoring is required; archived results and split lineage remain provenance.
