# New Gameplay Dataset With Static Validation

Use this workflow when adding a new gameplay workbook while preserving an
existing validation split for comparable metrics.

## Parameters

- `NEW_VERSION`: the new workbook and output version, such as `vN`.
- `BASE_VERSION`: the existing dataset version whose validation split should be
  kept fixed.

## Inputs

- Workbook remote: `googdrive:skyball/datasets/gameplay/${NEW_VERSION}.xlsx`
- Raw video remote: `googdrive:skyball/raw_video`
- Local workbook root: `/mnt/t/data/vball/skyball/gameplay/ground_truth`
- Local raw video root: `/mnt/t/data/vball/skyball/raw_video`
- Dataset root: `/mnt/t/data/vball/skyball/gameplay`
- Training output root: `/mnt/t/output/trn/skyball_gameplay`
- Static validation source: `/mnt/t/data/vball/skyball/gameplay/${BASE_VERSION}`

## Dataset Refresh Policy

Keep the validation split from `BASE_VERSION` static until the user explicitly
asks to re-split. Preserve all source videos that are marked `split=val` in the
base dataset metadata as validation in the new dataset. Source videos that are
new in `NEW_VERSION` should be added to `split=train` unless the user gives a
different split policy.

This policy makes metrics comparable across dataset refreshes: validation
changes reflect model or training-data changes, not a new validation
distribution. If the user requests an expanded validation cohort, retain the
base validation as a separately reported frozen slice, hold the expanded source
videos out of training, and report both the frozen slice and combined
validation population. Bind the added sources and their sampling policy in a
versioned manifest.

## Procedure

1. Sync the workbook:

```bash
mkdir -p /mnt/t/data/vball/skyball/gameplay/ground_truth
rclone copy googdrive:skyball/datasets/gameplay/${NEW_VERSION}.xlsx \
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
6. Make the review clips available to the user. Treat this as non-blocking when
   the user has explicitly requested an end-to-end dataset build; continue while
   preserving the review artifacts and manifest for later inspection.
7. Extract the versioned dataset under
   `/mnt/t/data/vball/skyball/gameplay/${NEW_VERSION}`:
   - use `train` and `val` split names,
   - store JPEGs in class folders,
   - write `metadata/windows.jsonl`, `metadata/images.jsonl`,
     `metadata/summary.json`, and `metadata/class_to_idx.json`,
   - include `source_video`, `raw_video_path`, `frame_paths`, `times_sec`,
     `class_name`, `binary_label`, boundary metadata, and `split`.
8. Verify every referenced image exists, frozen validation source videos were
   preserved, new source videos followed the requested split policy, and
   class/source-video counts are reported in `summary.json`.

## Training

Use this workflow only to define the dataset refresh and split policy. Training
experiments should be specified by the user request or by the training skill.
For the current best temporal model, the repo launcher is:

```bash
DSET_DIR=/mnt/t/data/vball/skyball/gameplay/${NEW_VERSION} \
  scripts/train_best_temporal_resnet50.sh
```

Use versioned output directories and do not overwrite existing runs. Compare
against `model_registry.yaml` on the same frozen validation split. Update
`model_registry.yaml` only if a run improves the selected validation metric.

## Notes

- Keep generated clips, datasets, checkpoints, predictions, and logs under
  `/mnt/t/data` or `/mnt/t/output`; do not commit them to this repo.
- If the user asks for additional experiments, such as a different input
  resolution or architecture, keep those experiment details in the run notes or
  command line rather than encoding them in this static-validation workflow.
