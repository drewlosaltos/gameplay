# Qwen gameplay encoder pilot — September 18, 2026

The user requested trying the Qwen vision encoder for gameplay after its jersey
results. Initially scheduled after the jersey portrait study, this run starts
in parallel on GPUs **0,1** following the user's explicit allocation of those
GPUs. Jersey square training and its portrait successor retain GPUs **2,3**.

Run: `/mnt/t/output/trn/skyball_gameplay/qwen36_frozen512_two_stage_20260918_r1`

Tmux: `qwen_gameplay_frozen_20260918_r1`. Immutable controller: `control_r1.py`.
The run uses a copied source tree, with SHA-256 hashes and all settings in
`config.json`. Commands, phase status and logs are recorded in the run root.
Source launcher: `scripts/run_qwen_frozen_experiment.py`.

## Model and training

- Original pretrained Qwen3.6-35B-A3B-FP8 vision tower, whose visual weights
  are BF16. No jersey fine-tuning weights are reused.
- Frozen 446,571,248-parameter encoder; 791,682 learned head parameters.
- Four full frames at 1 Hz, resized to 512×512 with the existing gameplay
  bilinear transform; RGB mean/std 0.5. Spatial mean of merged visual tokens
  followed by temporal mean; dropout 0.1 and LayerNorm → 384 → GELU → 2 head.
- Shared horizontal flips and color jitter 0.1; no crops, RandAugment or erasing.
- AdamW, head LR 3e-4, weight decay 1e-4, seed 20260916. Eight windows per
  GPU with accumulation 8 gives effective batch 128. BF16 training, FP32
  validation, TF32 disabled. Encoder stays in evaluation mode while training.
- Two legacy-v5 epochs, LR reduced after epoch 1, select by legacy validation
  F1; then four human-only epochs initialized from that best checkpoint, LR
  reduced after epoch 2, select by human validation F1.

This is an initial frozen-feature pilot. It preserves the current data,
frame sampling, resolution, augmentation and staged schedule. It changes the
encoder and head, and uses head LR 3e-4 rather than the baseline's 2e-5 because
the Qwen head starts randomly. The SigLIP baseline has earlier gameplay
fine-tuning history. Therefore this is not an isolated encoder ablation.

## Data and validation

Existing approved production datasets are reused without modification:

| Dataset | Train windows | Validation windows | Train / validation videos |
|---|---:|---:|---:|
| legacy v5 | 105,861 | 15,800 | 328 / 27 |
| human_app_v0 | 9,889 | 1,420 | 45 / 5 |

Preflight checked all 531,880 referenced image paths, binary labels,
four-frame timing, and source-video split isolation, including across datasets.
No missing images or train/validation video overlap. Manifest hashes are in
`preflight.json` and rechecked before training.

The selected human-stage checkpoint is evaluated on both held-out sets with
per-video metrics, original-category confusion, threshold sweeps and predictions.
The current SigLIP baseline has human F1 **96.0175%**, legacy F1 **93.8665%**
and combined F1 **94.0839%** at classification threshold 0.5.

Then the candidate scores the canonical ten full matches with exact pipeline
RGB preprocessing in FP32. The August production suite's bindings are checked
against the frozen primary-video/GT contract, including the two transcodes.
Compare at threshold **0.15**, padding **10 seconds per side**, merge gap **0**.
Report per-match and pooled confirmed events excluded, extra admitted footage
as a fraction of total duration, full event/rally-window retention and runtime.
Current SigLIP baseline: **0/8,848 events excluded**, **46.4249% extra footage**.
The baseline report labels this current model `candidate_selected_policy`;
its older `production_selected_policy` is not the baseline for this comparison.

No production promotion is part of this pilot. A recommendation would also
require the existing target-FP safety/effectiveness gate. Classification alone
does not complete the experiment: the controller only writes successful
completion after classification, full10 evaluation and result recording.

Completed evidence will appear in `result.json` and `REPORT.md` in the run root,
and `validation/qwen36_frozen512_two_stage_20260918_r1.json` in this repository.
The result explicitly compares each F1 with the prior baseline.

## Verification before launch

- Full gameplay test suite: **55 passed**.
- Real Qwen GPU smoke: finite BF16 head gradients, no encoder gradients,
  and exactly matching FP32 direct-window vs cached-frame scoring.
- Static compilation of controller; dataset and full10 input preflight passed.
- Both GPUs 0,1 idle before launch. No changes to the jersey controllers.

No gameplay validation result is available at launch.

## User-requested pivot

Stopped during legacy epoch 2 to switch to full fine-tuning with preserved aspect ratio. Epoch-1 legacy F1 was 89.5367%; no human-stage or full10 result. Checkpoints retained; the best frozen checkpoint initializes the replacement. See `qwen_unfrozen_wide_20260918.md`.
