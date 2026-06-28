# New Gameplay Dataset With Static Validation

Use this workflow when a new gameplay workbook appears at
`googdrive:skyball/datasets/gameplay/<version>.xlsx` and the goal is to add new
matches to training while keeping the validation set fixed for comparability.

## Inputs

- Workbook remote: `googdrive:skyball/datasets/gameplay/<version>.xlsx`
- Raw video remote: `googdrive:skyball/raw_video`
- Local workbook root: `/mnt/t/data/vball/skyball/gameplay/ground_truth`
- Local raw video root: `/mnt/t/data/vball/skyball/raw_video`
- Dataset root: `/mnt/t/data/vball/skyball/gameplay`
- Training output root: `/mnt/t/output/trn/skyball_gameplay`
- Current static validation source: `/mnt/t/data/vball/skyball/gameplay/v1`

## Dataset Refresh Policy

Keep the validation split static until the user explicitly asks to re-split.
For a new version such as `v2`, preserve the source videos already in
`v1/metadata/windows.jsonl` with `split=val` as validation. New source videos
from the new workbook should be added to `split=train` unless they are already
part of the frozen validation source-video set.

This policy makes v1 and v2 metrics comparable: validation changes indicate
model/data changes, not a new validation distribution.

## Procedure

1. Sync the workbook:

```bash
mkdir -p /mnt/t/data/vball/skyball/gameplay/ground_truth
rclone copy googdrive:skyball/datasets/gameplay/<version>.xlsx \
  /mnt/t/data/vball/skyball/gameplay/ground_truth/
```

2. Parse the workbook with pandas/openpyxl. Discover sheet names, columns,
   labels, source-video IDs, and time formats before assuming schema details.
3. Resolve every source video against `/mnt/t/data/vball/skyball/raw_video`.
   Copy missing videos from `googdrive:skyball/raw_video` while preserving remote
   subdirectories.
4. Validate local raw video durations against each workbook video's maximum
   `end_sec`. Treat mismatches over about 0.25 seconds as blockers.
5. Create review clips from raw videos before full extraction:
   - about 12 four-second clips per selected source video,
   - balanced across gameplay and non-gameplay labels,
   - no samples within 30 seconds of label boundaries unless the user overrides,
   - overlay source video, timestamp, and label.
6. Stop for user approval of review clips.
7. Extract the versioned dataset under
   `/mnt/t/data/vball/skyball/gameplay/<version>`:
   - use `train` and `val` split names,
   - store JPEGs in class folders,
   - write `metadata/windows.jsonl`, `metadata/images.jsonl`,
     `metadata/summary.json`, and `metadata/class_to_idx.json`,
   - include `source_video`, `raw_video_path`, `frame_paths`, `times_sec`,
     `class_name`, `binary_label`, boundary metadata, and `split`.
8. Verify every referenced image exists, split source videos are disjoint except
   for intentionally frozen validation carry-over, and class/source-video counts
   are reported in `summary.json`.

## Training Experiments

Train the current best architecture first:

```bash
DSET_DIR=/mnt/t/data/vball/skyball/gameplay/<version> \
  scripts/train_best_temporal_resnet50.sh
```

The launcher defaults to the current best recipe: pretrained ResNet-50, four
frames, average pooled embeddings, binary head, 224x224 transform, cosine LR.

Run the higher-resolution comparison by overriding the transform size:

```bash
DSET_DIR=/mnt/t/data/vball/skyball/gameplay/<version> \
IMAGE_HEIGHT=512 IMAGE_WIDTH=512 \
  scripts/train_best_temporal_resnet50.sh
```

Use versioned output directories and do not overwrite existing runs. Compare
against `model_registry.yaml` on the same frozen validation split. Update
`model_registry.yaml` only if a run improves the selected validation metric.

## Notes

- The extracted source images for v1 were 512 pixels on the long edge, but model
  training transforms are independent. A 512x512 experiment means setting the
  model transform to 512x512.
- Keep generated clips, datasets, checkpoints, predictions, and logs under
  `/mnt/t/data` or `/mnt/t/output`; do not commit them to this repo.
