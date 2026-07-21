# Gameplay v5 Dataset and Validation Protocol

This workflow is the reproducible contract for building and evaluating the
SkyBall gameplay v5 dataset. The versioned source-of-truth split manifest is
`validation/gameplay_v5_split_v1.json`.

## Source Data

- Workbook: `/mnt/t/data/vball/skyball/unified_staging/gameplay/v5.xlsx`
- Workbook SHA-256:
  `a2c02b87b434b2a8cd8f24f132f83b4bed6627f35b2491f73c6a16106afe1961`
- Raw videos: `/mnt/t/data/vball/skyball/raw_video`
- Output dataset: `/mnt/t/data/vball/skyball/gameplay/v5`
- Frozen validation metadata: `/mnt/t/data/vball/skyball/gameplay/v2/metadata`

## Split Contract

The 355 source matches are split by source video, never by frame or window.

- `v2_fixed`: preserve the seven v2 validation matches and their exact legacy
  sampling. This slice provides the historical regression metric.
- `v5_expanded`: hold out the 20 matches named in the split manifest. The
  selection covers v3 youth, v4 college, and v5 college cohorts across gameplay
  fraction, duration, and transition-rate strata.
- `train`: all remaining 328 matches. Neither validation cohort may appear in
  training.

The combined `val` directory contains both validation cohorts. Every validation
record carries `validation_cohort` so evaluation can report `v2_fixed`,
`v5_expanded`, and combined metrics separately.

## Sampling Contract

Every sample is a four-frame window with frames at one-second intervals.

| Population | Gameplay start stride | Non-gameplay start stride |
|---|---:|---:|
| v5 training | 20 seconds | 10 seconds |
| frozen v2 validation | 20 seconds | 1 second |
| expanded v5 validation | 10 seconds | 10 seconds |

Expanded validation uses class-independent temporal sampling so its class mix
tracks the source-video timeline. Training is sparsified to prevent long
non-gameplay regions in college matches from dominating the dataset. Frozen v2
validation is not resampled.

Images are extracted at 1024 pixels on the long edge with JPEG quality 88.
Transition exclusion zones remain 20 seconds around pre-match/gameplay changes
and 5 seconds around all other label changes.

## Build Command

```bash
python scripts/build_gameplay_v0_dataset.py extract \
  --workbook /mnt/t/data/vball/skyball/unified_staging/gameplay/v5.xlsx \
  --raw-root /mnt/t/data/vball/skyball/raw_video \
  --output-root /mnt/t/data/vball/skyball/gameplay/v5 \
  --static-val-metadata /mnt/t/data/vball/skyball/gameplay/v2/metadata \
  --expanded-val-manifest validation/gameplay_v5_split_v1.json \
  --gameplay-stride-sec 20 \
  --negative-stride-sec 1 \
  --train-negative-stride-sec 10 \
  --expanded-val-stride-sec 10 \
  --long-edge 1024 \
  --jpeg-quality 88 \
  --extract-workers 4
```

Extraction may decode four independent source videos concurrently. Worker
count affects wall-clock and temporary storage only; it does not change sample
membership, timestamps, resize, or JPEG settings.

Do not use `--force` on an existing v5 output without exact-path approval. The
generated `metadata/summary.json` must reproduce the split and sampling policy
before training begins.

## Evaluation

Train on `split=train` and select checkpoints on the combined v5 validation
population. After training, report the same checkpoint on:

1. `v2_fixed`, for direct comparison with the old network;
2. `v5_expanded`, for new-cohort generalization;
3. combined validation, for the overall v5 operating point.

Calibrate the deployment threshold on combined validation while retaining the
frozen-v2 score as a non-regression result. The separate ten-match full-video
gameplay gate remains the final operational validation.
