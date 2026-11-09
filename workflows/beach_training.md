# Beach gameplay training

Moved from `README.md` on 2026-11-08. This is still the recipe behind the beach
production model (`domain_models.beach` in `model_registry.yaml`).

## Recipe

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

