# SkyBall gameplay classification

This repository supports both indoor and beach gameplay training on `main`.

The canonical indoor model is **Qwen3.6 vision, 672x384, single-stage FP8**.
The production checkpoint is `v8_val48_qwen672x384_single_stage_fp8_20261006_r3`,
selected at epoch 3 of four epochs: clean val48 F1 **96.1224%** on 16,281 windows
(previous production: 94.6217% on the same windows).

## Canonical training

Use `scripts/train_qwen_canonical.py`. Initialize the vision encoder from the
pretrained Qwen snapshot, with a fresh gameplay head and no gameplay checkpoint.
Train the entire model at 672x384 for four epochs. Four frames at 1 Hz share
one encoder; their embeddings are averaged before a binary MLP. Use gray128
letterboxing, shared horizontal flips and +/-5-degree rotation, color jitter
0.1, temporal dropout 0.1, and no random crop, RandAugment or erasing.

The exact recipe is in `config/indoor_canonical.json`: four GPUs, per-GPU batch
4, global/effective batch 16, AdamW head LR 1e-4 and encoder LR 3e-6, weight
decay 1e-4, half-epoch warmup, step decay every two epochs by 0.1, gradient
clipping 1.0. Tensorwise TorchAO FP8 linears and `torch.compile(mode="default")`
accelerate training under BF16 AMP; validation and published inference use FP32.

Use the `skyball-prod-infer` environment and TorchAO 0.17.0+cu128. The validated local
TorchAO overlay is `/mnt/t/output/skyball_gameplay/v8_qwen_fp8_prototype_20261006_r1/site`.
The launcher checks dependencies and records their versions and input identity.

```bash
PYTHONPATH=/mnt/t/output/skyball_gameplay/v8_qwen_fp8_prototype_20261006_r1/site:$PWD \
  conda run --no-capture-output -n skyball-prod-infer \
  python scripts/train_qwen_canonical.py \
  --output-dir /mnt/t/output/trn/skyball_gameplay/<new-versioned-run> \
  --gpus 0,1,2,3 --dry-run
```

Remove `--dry-run` after checking the plan and GPU availability; run long jobs
in tmux. **Clean val48 is the default**, frozen by source video in
`validation/gameplay_indoor_v8_val48_clean_v2.json`. No val20/val50 reruns are
required. Train from `metadata/windows.jsonl`: this overlay's image directories
contain excluded JPEGs, so do not use ImageFolder. The launcher rejects changed
validation membership, overlap, and the known reencoded training duplicates.
Model-assisted validation labels await human acceptance; F1 remains provisional.

Every selected checkpoint also requires the canonical full10 gate. The selected
production policy is **threshold 0.9, padding 5 seconds per side, merge gap 0,
1 Hz sampling**. It retains all 8,848 events and 1,370 rallies and rejects
94/114 target false positives. Outside-event footage is 39.6748% of total match
duration, versus 45.1950% for previous production at its 0.5/20-second policy.
These ten matches informed policy selection and are retrospective validation.
See `workflows/qwen_v8_production_20261007.md` for evidence and rollback.

The old v5 launcher and two-stage recipes below are historical reproductions.

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
   clean val48, comparing production/candidate on identical samples. Report the selected epoch, not just the last epoch.
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

Current production policy: threshold **0.9**, **5 seconds per side**, gap 0.
This is separate from the 0.5 classifier threshold and travels with the model.
Production requires portable/runtime qualification and pipeline acceptance;
a high classification F1 alone is insufficient.

See [training progression](workflows/human_gameplay_finetune_v0.md),
[full-video validation](workflows/gameplay_gate_validation.md), and contracts
in `validation/`. Published bundles must include dataset locations/hashes,
split membership, training lineage and commands, selected epochs, both
validation sources, and limitations. Model-assisted labels are not verified
untouched LLM output.
