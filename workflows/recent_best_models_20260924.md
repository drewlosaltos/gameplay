# Gameplay reboot handoff — September 24, 2026

## Best recent indoor classifiers

Both models below use the full v3 303-match snapshot: **283 training matches / 69,897 four-frame windows**, and the same **20 validation matches / 4,828 windows**. Metrics use **coarse gameplay labels at threshold 0.5**, not detector-derived rally labels. Pre-match FP is included in total FP.

| Model | Selected epoch | Coarse F1 | Gameplay recall | Non-gameplay recall | FN | FP | Pre-match FP / 504 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen 672×384, best classifier F1** | 4 of 4 | **96.110%** | 95.570% | 94.454% | 134 | 100 | 27 |
| **SigLIP-Base 512×512, best non-Qwen** | 5 of 8 | **95.812%** | 97.190% | 90.460% | 85 | 172 | 53 |

Qwen rejects more non-gameplay but misses more coarse gameplay than SigLIP. Classifier F1 alone does not qualify either model for production: full-video gate validation was explicitly deferred. Neither result authorizes production integration.

### Qwen

- Run: `human_app_v3_val20_qwen672x384_rotate5_e4_4gpu_b8_20260922_r1`
- Best checkpoint: `/mnt/t/output/trn/skyball_gameplay/human_app_v3_val20_qwen672x384_rotate5_e4_4gpu_b8_20260922_r1/best_qwen672x384_e4.pth.tar`
- SHA256: `be8bcc7f7411ae24c0adaa69674f20b1e7f6ad9a17decc247e289e0d4138f3f2`
- Recipe: fully unfrozen Qwen vision encoder; four RGB frames at 1 Hz; 672 wide × 384 high, aspect-preserving letterbox; shared ±5° rotation, flips and light color jitter. BF16 training / FP32 validation; global batch 8 on four GPUs (2/GPU), no accumulation. Head LR 1e-4 / encoder LR 1e-6, half-epoch warmup, 10× LR drop after epoch 2; four epochs.
- Weak-data initializer: `/mnt/t/output/trn/skyball_gameplay/qwen36_unfrozen_wide352x192_two_stage_20260918_r1/legacy_adaptation/best_qwen_unfrozen_wide.pth.tar`
- Code: `/mnt/t/output/skyball_gameplay/v3_recipe_screen_20260922_r1/encoder_code`, commit `e8d77785e5c1f2816dea48a7f93fc799fc980923`.
- Required transformer overlay in the saved launch config: `/mnt/t/output/trn/jersey_recognition/qwen36_vision_indoor_v10_20260918_r2/runtime`. Qwen snapshot path and complete arguments are in the run's `config.json`; use those for reproduction.
- Workflow/evidence: `/mnt/t/output/skyball_gameplay/human_app_v3_qwen672x384_full4gpu_b8_20260922_r1/`; completed successfully. `classification_comparison.json` compares the two models above on identical val20 records.
- Epoch F1: **95.353%, 95.909%, 95.817%, 96.110%**. Earlier discussion naming epoch 2 as best is superseded. The `best_` checkpoint now contains **epoch 4**; epoch-2 predictions/metrics remain, but no separate epoch-2 checkpoint is retained in this run directory.
- No Google Drive publication is recorded for this Qwen run.

### SigLIP

- Run: `human_app_v3_val20_siglip512_stage2_fresh8_step4_b128_20260921_r1`
- Best checkpoint: `/mnt/t/output/trn/skyball_gameplay/human_app_v3_val20_siglip512_stage2_fresh8_step4_b128_20260921_r1/best_siglip512_app_v3_val20_fresh8_step4_b128.pth.tar`
- SHA256: `24e354ba1b451bf75df884a0311ea12127d3ace3b55bb49d08be00957b1d6c85`
- Recipe: shared SigLIP-Base encoder, four RGB frames at 1 Hz, full frame stretched to 512×512, flips/light color jitter, no rotation. Fresh weak-data initialization; head LR 2e-5 / encoder LR 1e-6, 10× drop after epoch 4; eight epochs, effective batch 128.
- Weak-data initializer: `/mnt/t/output/trn/skyball_gameplay/v5_siglip512_fullcourt_adapt_20260916/best_siglip512_legacy_fullcourt_adapt.pth.tar`
- Code/evidence: `/mnt/t/output/skyball_gameplay/human_app_v3_fresh8_20260921_r1/`; code commit `5a55096`.
- **Uploaded and checksum-verified:** [Google Drive model package](https://drive.google.com/drive/folders/1ntEFSrmBDpBcAGnTRMqjDaKYcIka7tzM).
- Remote: `googdrive:skyball/checkpoints/gameplay/human_app_v3_siglip512_coarse_val20_fresh8_20260922_r1`. Its publication name differs from the training run name; it is the same checkpoint. Contains 25 files including weights, configuration, source archive, and validation evidence. Last remote recheck September 22: 25 matching, zero differences.
- Local package/receipt: `/mnt/t/output/skyball_gameplay/model_upload_v3_coarse_20260922_r1/`.

## Recent ablations and decisions

Mini studies used 50 training matches / 10,462 windows and the unchanged val20. Best F1: SigLIP512 reference 93.51%; rotation ±5° 93.33%; letterbox768×384 92.83%; square768 93.55%; four adjacent30Hz frames + rotation 92.59%; Qwen672×384 + rotation 94.70%. Encoder/pretraining/batch differences mean these are recipe comparisons, not isolated backbone effects. Full evidence: `/mnt/t/output/skyball_gameplay/v3_recipe_screen_20260922_r1/RESULTS.md`.

- Retain shared ±5° rotation for subsequent experiments, per user instruction. It improved mini recall but increased false positives.
- Four adjacent30Hz frames shorten context to ~0.1s; they underperformed four1Hz frames. Dense sampling over the same three-second context has **not** been tested.
- Small production DINOv2: three adjacent30Hz grayscale channels at native768, 22.5M encoder parameters, fresh classifier. **Stopped by user; do not auto-resume.** Epoch1 coarse F187.18%. BF16 resolved initial FP16 loss-scale overflow; checkpoint-wrapper and three-frame prediction-export bugs were fixed. Code at `v3_recipe_screen_20260922_r1/small_dino_code`, commit `2d5d311`; partial checkpoints in `human_app_v3_tiny50_val20_dino_small768_gray3_30fps_e4_20260922_r3` under the training root.
- Increasing Qwen global batch4→8 on four GPUs improved sustained training throughput25.3→34.8 windows/s (+37%). Keep batch8/noaccum for reproducing the completed run.
- Potential follow-ups, not launched: 768+rotation together; random sampling-offset augmentation retaining1Hz context; full-video gate qualification if requested.

## Reboot / resume notes

- **Gameplay Qwen training is complete; no recent gameplay trainer needs resuming.** Prior two-GPU/globalbatch4 attempts are intentionally stopped and preserved. DINO is intentionally stopped.
- **Other work is active:** at this handoff all four GPUs are running the separate detector study configured by `/mnt/t/output/rf-detr-mf/studies/cnf_r1280_20260921_r1/production_r1280_a4.json` (worker PIDs1301600–1301603 at inspection). A reboot will interrupt it; coordinate checkpoint/stop with that work's owner. This handoff does not stop or assess resumability of that detector run.
- An older gameplay controller remains alive: `/mnt/t/output/skyball_gameplay/human_app_v2_1_rally_gtgated_20260921_r2/run_tiny_first_workflow.py`. Its status is stale at `v2_1_finegrain_training`; do not treat that as an active trainer or restart it automatically. Finegrain/weak453 expansion and full-video evaluations remain deferred.
- Dataset: `/mnt/t/data/vball/skyball/gameplay/human_app_v3_val20_20260921`; manifest SHA256 `54d82f8ffe09a3b23e4b80eff1ee529aa7b0aab7da78ab6844f78441d3c5d07f`. Frozen303 snapshot was explicitly approved while Drive export was in progress; not all labels were independently human accepted.
- Main checkout is on `main` but **has outstanding tracked/untracked experimental changes**. Do not reset/clean it. Recent study implementations are committed in separate worktrees under `/mnt/t/output/skyball_gameplay`; keep those directories. Existing registry entries are not being changed into production recommendations by this handoff.
- Logs, checkpoints, TensorBoard events, and HTML review files are on disk and survive reboot; tmux/HTTP/TensorBoard processes do not. Gameplay TensorBoard can be restarted with `conda run --no-capture-output -n pt5090new tensorboard --logdir /mnt/t/output/trn/skyball_gameplay --port 6008 --bind_all`.
- Coarse FN review: `/mnt/t/output/skyball_gameplay/v3_best_coarse_fn85_20260922_r1/` (85 cases). Detailed three-case sampling analysis: `/mnt/t/output/skyball_gameplay/v3_best_fn_neighborhoods_20260922_r1/`. Sampling phase was a strong sensitivity in FN002; these are analysis artifacts, not corrected ground truth.
