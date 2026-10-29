# Unfrozen Qwen gameplay at approximately 256-square pixel budget

The user requested this lower-resolution pivot after seeing the 672×384
run's roughly 85–90-minute legacy training epochs. That larger run was stopped
before its first full validation, with all outputs retained.

Run: `/mnt/t/output/trn/skyball_gameplay/qwen36_unfrozen_wide352x192_two_stage_20260918_r1`.
Tmux: `qwen_gameplay_wide256_20260918_r1`. GPUs **0,1**.
The immutable `control_r1.py` executes hashed source snapshots. Jersey portrait
fine-tuning continues independently on GPUs 2,3.

## Resolution and recipe

- **352 wide × 192 high**, 67,584 pixels: 3.125% above 256² and 73.81% fewer
  pixels than the superseded 672×384 experiment. Original 16:9 frames resize
  to approximately 341×192, with centered RGB-128 side padding. No cropping
  or stretching to the canvas ratio. Four frames at 1 Hz remain.
- Initialize from the preserved frozen-square gameplay best legacy epoch-1
  checkpoint (89.5367% legacy F1), not a smoke checkpoint. Unfreeze the whole
  Qwen vision tower and learned gameplay head.
- AdamW head LR 1e-4, vision LR 1e-5, weight decay 1e-4, half-epoch warmup,
  clip gradient norm at 1, BF16 training, FP32 validation, TF32 disabled.
- Eight windows per GPU, accumulation 8, effective batch 128. DDP gradient
  synchronization occurs at optimizer updates.
- Two legacy-v5 epochs, select by legacy F1, then four human-only epochs,
  select by human F1. Step LR after epoch 1 / epoch 2 respectively, factor 0.1.
  Shared flips/color jitter 0.1, dropout 0.1, seed 20260916 unchanged.

This retains the approved data and validation definitions documented in
`qwen_unfrozen_wide_20260918.md`. Before each run, recheck dataset manifest
hashes and code/initializer identities. The previous full preflight found
531,880 image references present and no train/validation video overlap.
Final evaluation includes human and legacy classification plus the exact
canonical full10 videos and GT, at threshold 0.15, 10-second padding per side,
merge gap 0. Full-video decoding uses the same 1 Hz timing, at long edge 352,
then the checkpoint's aspect-preserving geometry and normalization.

Validation evidence is written automatically to the run root and
`validation/qwen36_unfrozen_wide352x192_two_stage_20260918_r1.json`.
No production promotion. Tiny bounded smoke validation scores are not model
results. The real two-GPU smoke verifies a full effective-batch128 update,
FP32 validation, checkpoint metadata and that encoder parameters changed.

## Historical resolution evidence

The July 19 SigLIP study compared native **224×224** and **512×512** model
variants on the same 15,800 v5 validation windows. At threshold 0.5, F1 was
92.8808% versus 93.9879% (+1.1071 percentage points); accuracy was 92.5253%
versus 93.8228%. Both used four frames and matching augmentation/LR settings,
but different pretrained resolution variants and selected epochs. It does
not establish Qwen's resolution tradeoff or an exact 256-versus-512 effect.
The new run is a practical lower-compute experiment, not a completed matched
Qwen resolution comparison.

## Stopped at user request

All two legacy and four human training epochs completed. Best legacy-stage F1 was 92.9318% (epoch 1). Best human F1 was 87.8338% (epoch 2), versus production 96.0175%; its 0.5-threshold confusion was TP 740, FN 173, FP 32, TN 475. A diagnostic threshold sweep on the same human validation raised F1 to 94.0299% at threshold 0.00905; this is in-sample threshold selection, not an independent result.

The user stopped the run during final legacy evaluation. Human evaluation completed; final legacy and full10 validation are incomplete. All checkpoints and outputs are retained, GPUs 0,1 released, no promotion. See the run root `stopped_by_user.json` and repository `validation/qwen36_unfrozen_wide352x192_two_stage_20260918_r1_stopped.json`.
