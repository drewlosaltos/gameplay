# SkyBall gameplay classification

The canonical indoor recipe is **two-stage, full-frame SigLIP-Base**: adapt an
existing gameplay checkpoint on the older dataset, then fine-tune exclusively
on human-labeled data. The encoder has about 93.5M parameters, shared across four
frames sampled at 1 Hz; average embeddings feed a binary head.

## Canonical training

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
