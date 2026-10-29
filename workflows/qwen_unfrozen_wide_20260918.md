# Unfrozen Qwen gameplay with preserved aspect ratio

The user explicitly pivoted from frozen square gameplay to full fine-tuning
with preserved native aspect ratio. The frozen pilot was stopped during legacy
epoch 2. Its best epoch-1 checkpoint and **89.5367% legacy-validation F1** are
retained. That stopped run did not complete the human stage or full10 evaluation.

New run: `/mnt/t/output/trn/skyball_gameplay/qwen36_unfrozen_wide672x384_two_stage_20260918_r1`.
GPUs **0,1**; jersey portrait fine-tuning remains on GPUs 2,3.
Tmux: `qwen_gameplay_wide_20260918_r1`. Immutable `control_r1.py` executes
the hashed `code_snapshot`; original stopped control files remain intact.

## Geometry and initialization

One extracted frame from each of 355 legacy and 50 human source videos was
checked: every sampled frame is **1024×576 (16:9)**. Resize to **672×378**
without stretching, then pad three pixels on top and bottom with RGB 128 for
a **672×384** tensor. The canvas has 258,048 pixels, 1.5625% fewer than 512².
Other aspect ratios use the same centered fit-and-pad operation without crop.
Geometry is checkpoint metadata and is honored by dataset evaluation and the
full-video scorer. Four frames at 1 Hz and spatial/temporal mean pooling remain.

Initialize from the stopped frozen gameplay run's best legacy epoch-1
checkpoint. Its encoder is still the original pretrained Qwen tower; its MLP
head has learned gameplay. All encoder and head parameters are trainable.
This changes both fine-tuning and geometry and is not an isolated aspect-ratio
ablation.

## Schedule and validation

- AdamW: head LR 1e-4, encoder LR 1e-5, weight decay 1e-4; half-epoch linear
  warmup and gradient clipping at norm 1 for the newly unfrozen encoder.
- Two legacy epochs (step LR after one), then four human-only epochs (step
  after two), selecting each stage by its validation F1. Stage two initializes
  from the best legacy checkpoint and resets optimizer/scheduler.
- Two windows per GPU, accumulation 32, effective batch 128. DDP synchronizes
  at optimizer updates. BF16 mixed precision training, FP32 validation, TF32
  disabled. Shared flips/color jitter 0.1, dropout 0.1, seed 20260916.
- Identical approved datasets and held-out source videos. The prior preflight
  checked all 531,880 image references and cross-dataset split isolation;
  manifest hashes are verified before the new run.
- Final human and legacy dataset classification plus canonical ten-match
  evaluation are automatic. Threshold 0.15, padding 10 seconds per side,
  merge gap 0; compare zero excluded events first, then extra footage.
- Full-video sampling uses the same pipeline 1 Hz timing and bilinear ffmpeg
  filter, now at long edge 672, followed by the checkpoint's letterbox geometry.
  The older baseline's scores retain its original 512-square preprocessing.
  Both use the exact same canonical videos, GT and temporal policy.

Current selected SigLIP baseline: human F1 96.0175%, legacy F1 93.8665%,
combined F1 94.0839%; full10 excludes 0/8,848 events with 46.4249% extra
footage / total duration. No production promotion is authorized by this pilot.

## Checks and artifacts

56 tests pass, including letterbox geometry and checkpoint metadata round-trip.
A two-GPU bounded real forward/backward, FP32 validation and checkpoint-save
smoke passed. An additional smoke uses the actual effective batch 128 and
gradient clipping. These tiny smoke validation metrics are not model scores.
The decoder smoke checks the candidate and production sampling counts agree.
Commands, initializer hash, code hashes and preflight live in the output root;
completed results go to its `result.json`/`REPORT.md` and the repository's
`validation/qwen36_unfrozen_wide672x384_two_stage_20260918_r1.json`.

## Superseded by the user

The 672x384 run was stopped before its first full validation after the user requested an aspect-preserving pixel budget near 256x256. Outputs and smoke artifacts remain. The replacement is documented in `qwen_unfrozen_wide256_20260918.md`.
