# SkyBall gameplay classification

This repository supports both indoor and beach gameplay training on `main`.

The canonical indoor recipe is the **Qwen3.6 vision 672x384 v5 fine-tune**.
Initialize from the selected Qwen legacy-adaptation checkpoint and fine-tune
the full four-frame model for four epochs on `human_app_v5_val20_20260925`.
Select the checkpoint by F1 on the frozen 20-video human validation split.
The September 25 run selected epoch 4 (F1 96.5858% on 4,828 windows).
The former two-stage SigLIP-Base recipe remains below for reproduction.

## Canonical training

Use `scripts/train_qwen_v5_canonical.py` in `skyball-prod-infer` (which has the
Qwen3.6 vision implementation). The recipe uses four frames at 1 Hz, a shared
unfrozen Qwen encoder, temporal average pooling, 384x672 aspect-preserving
letterbox with gray padding, shared horizontal flips and +/-5-degree rotation,
light color jitter, and no random crop, RandAugment, or erasing. On two GPUs,
global microbatch 4 with accumulation 2 gives effective batch 8. The encoder
learning rate is 1e-6 and the head learning rate is 1e-4.

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run --no-capture-output -n skyball-prod-infer \
  python scripts/train_qwen_v5_canonical.py \
  --output-dir /mnt/t/output/trn/skyball_gameplay/<new-versioned-run> \
  --gpus 2,3 --dry-run
```

Remove `--dry-run` after checking the plan, GPU availability, and reviewed
dataset. Validate every selected checkpoint on the frozen human and legacy
sets and the canonical ten full videos using
`workflows/gameplay_gate_validation.md`. The currently selected full-video
policy is threshold **0.50**, padding **20 seconds per side**, merge gap **0**,
at 1 Hz. This policy was selected on the same ten matches used for the
retrospective gate, so those matches are not an independent test.

### Previous SigLIP recipe

Use reviewed datasets split by source video. Keep older/model-assisted and
human labels separate. Validation videos are fixed; never split by frame.

| Setting | Legacy adaptation | Human-only fine-tuning |
|---|---|---|
| Initialization | Explicit existing gameplay checkpoint | Best legacy-stage checkpoint |
| Epochs run | 2 | 4 |
| Checkpoint selection | Legacy-validation F1 | Human-validation F1 |
| Head / encoder LR | 2e-5 / 1e-6 | 2e-5 / 1e-6 |
| LR reduction | 10x after epoch 1 | 10x after epoch 2 |
| Optimizer / weight decay | AdamW / 1e-4 | AdamW / 1e-4 |
| Effective batch | 128 | 128 |
| Input | Four full frames, 512x512 | Four full frames, 512x512 |

Resize the entire image to 512x512, stretching its aspect ratio. Use shared
horizontal flips and color jitter 0.1, dropout 0.1, and no random crop,
RandAugment or erasing. Train with AMP and validate in FP32. Seed: 20260916.
Source JPEGs retain 1024 pixels on the long edge.

Run in tmux using the SkyBall environment, after checking GPUs:

```bash
nvidia-smi
conda run --no-capture-output -n pt5090new python scripts/train_two_stage_fullframe.py \
  --legacy-data /mnt/t/data/vball/skyball/gameplay/v5 \
  --human-data /mnt/t/data/vball/skyball/gameplay/human_app_v0 \
  --init-checkpoint /path/to/approved_initializer.pth.tar \
  --production-suite /path/to/completed_canonical_full10_production_run \
  --output-root /mnt/t/output/trn/skyball_gameplay/new_versioned_run \
  --gpus 0,1,2,3
```

Add `--dry-run` to inspect the plan without jobs or output writes. One, two or
four GPUs are supported; accumulation preserves effective batch 128. The
launcher writes exact commands, input hashes, a code snapshot, stage logs,
classification comparisons and ten-match validation. It refuses an existing
output directory. Dataset approval and integrity validation are prerequisites;
the launcher does not grant production approval.

The selected September 16 indoor model adapted on **v5**, then fine-tuned on
`human_app_v0`. An approved newer legacy dataset such as v6 can be passed with
`--legacy-data`; it does not require a version-specific training script.

## Beach training

The beach recipe remains a separate four-epoch SigLIP-Base fine-tune from the
v5 indoor checkpoint. It uses four 512x512 frames, random crops, shared
RandAugment, and checkpoint selection by validation accuracy. The indoor
full-frame recipe has not been validated as a beach replacement.

For the existing approved v7 dataset, run the following inside tmux after
checking GPUs. Use a new `RUN_NAME` for each run; `DRY_RUN=1` previews the setup.

```bash
nvidia-smi
DSET_DIR=/mnt/t/data/vball/skyball/gameplay/v7 \
RUN_NAME=beach_siglip512_$(date +%Y%m%d_%H%M%S) \
TIMM_MODEL=vit_base_patch16_siglip_512.v2_webli \
IMAGE_HEIGHT=512 IMAGE_WIDTH=512 \
GPUS=0 WORKERS=8 TEMPORAL_BATCH_SIZE=16 GRAD_ACCUM_STEPS=8 EVAL_BATCH_SIZE=32 \
INIT_CHECKPOINT=/mnt/t/output/trn/skyball_gameplay/v5_siglip_base_f4_avg_h512w512_step4s2_lr1e4_blr5e6_20260719/best_siglip_base_f4_avg_binary_h512w512_step4s2_pre_drop010.pth.tar \
  scripts/train_siglip_temporal.sh
```

The launcher trains and writes classification metrics, predictions, and cohort
summaries. Preserve the 38-train/9-validation source split in
`validation/gameplay_v7_beach_population_v1.json` and
`validation/gameplay_v7_beach_val_v1.json`. For a new beach dataset, use the
shared builder and validator with explicit population and validation manifests.
`scripts/run_v7_beach_finetune.sh` preserves the original end-to-end v7 import
procedure; it requires a new dataset directory and authorization to continue
past review clips. It is not the entry point for retraining an existing dataset.

The July beach checkpoint scored **0.93365 F1** on 1,991 held-out windows,
versus 0.71702 for its initializer. Its recorded user-selected production scope
is beach only; full-video gate validation, portable runtime qualification, and
pipeline integration were not completed. Those historical labels do not qualify
a new candidate: new runs still require both validation sources below.

## Two required validation sources

1. **Classification:** F1, precision, recall and confusion at threshold 0.5 on
   human and older validation, comparing initializer/candidate on identical
   samples. Report the selected epoch, not just the last epoch.
2. **Ten full matches:** confirmed events excluded by the gate, and extra
   footage admitted as a percentage of the entire match. Extra footage is
   outside the union of +/-3-second windows around confirmed GT events.
   Pool seconds across matches because FPS differs. Prioritize zero excluded
   events, then compare extra admitted footage.

Canonical inputs:
`/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1`.
Preserve its two 30-fps transcodes and remapped GT. Reuse only verified
production artifacts. The scorer uses production RGB preprocessing; the
analyzer reconstructs saved production admission exactly. State threshold,
padding per side, merge gap and runtime. Compare identical settings separately
from selected policies. These ten matches are a second validation source, not
an independent test estimate after model/policy selection.

Current candidate policy: threshold **0.15**, **10 seconds per side**, gap 0.
This is separate from the 0.5 classifier threshold and travels with the model.
Production requires portable/runtime qualification and pipeline acceptance;
a high classification F1 alone is insufficient.

See [training progression](workflows/human_gameplay_finetune_v0.md),
[full-video validation](workflows/gameplay_gate_validation.md), and contracts
in `validation/`. Published bundles must include dataset locations/hashes,
split membership, training lineage and commands, selected epochs, both
validation sources, and limitations. Model-assisted labels are not verified
untouched LLM output.
