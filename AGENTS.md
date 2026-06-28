# SkyBall Gameplay

This repo houses the gameplay classification project: deciding whether frames or short windows from SkyBall volleyball videos represent active gameplay.

## Source Data

- Label workbook remote: `googdrive:skyball/datasets/gameplay/<version>.xlsx`
- Local workbook root: `/mnt/t/data/vball/skyball/gameplay/ground_truth`
- Raw video root: `/mnt/t/data/vball/skyball/raw_video`
- Raw video remote: `googdrive:skyball/raw_video`
- Extracted gameplay dataset root: `/mnt/t/data/vball/skyball/gameplay`
- Output root: `/mnt/t/output/skyball_gameplay`
- Training root: `/mnt/t/output/trn/skyball_gameplay`

Gameplay workbook timestamps refer to full raw match videos, not the
`unified_dataset/images_full` play clips. Do not use `images_full` to interpret,
review, or extract gameplay classification samples from this workbook.

The first Google Drive workbook is named `v1.xlsx`; old local `v0` artifacts are
legacy names for that first dataset. For new workbooks such as `v2.xlsx`, use
the Google Drive version name in local outputs.

## Workflow

1. Inspect `gameplay.xlsx` and map each source video to a full raw match video
   under `/mnt/t/data/vball/skyball/raw_video`.
   - If a raw match is missing locally, rclone it from `googdrive:skyball/raw_video`
     while preserving the remote subfolder layout.
   - Confirm raw video duration matches the workbook's maximum `end_sec` before
     trusting extracted timestamps.
2. Build review clips before making the full dataset:
   - Select a few source videos.
   - Sample 12 four-second regions per source video.
   - Use a balanced mix of gameplay and non-gameplay classes when available.
   - Render directly from raw match videos, never from `images_full`.
   - Avoid ambiguous samples around labeled boundaries using the boundary policy below.
   - Overlay source video name, class, and timestamp.
   - Concatenate the 12 samples into one review video per source video.
3. Stop for user review before extracting the full training dataset.
4. After approval, extract an ImageNet-style frame dataset under
   `/mnt/t/data/vball/skyball/gameplay/vN`.
5. Train in stages:
   - Current best: pretrained ResNet-50, four-frame, four-second window model,
     average pooled encoder embeddings, binary classification.
   - Higher-resolution ablation: repeat the temporal model with a 512x512 model
     transform.
   - Baseline: ResNet-50, single image, binary classification.
   - Optional: binary or multiclass heads for either image or window inputs.

For new gameplay versions, read `workflows/new_dataset_static_val.md`. The
current comparison policy is to keep the validation source videos from `v1`
static and add new matches to the training split unless the user asks to
re-split validation.

## Dataset Contract

For new versions, extract actual JPEG images from raw videos into ImageNet-style folders:

```text
/mnt/t/data/vball/skyball/gameplay/vN/
  train/
    gameplay/
    timeout/
    setbreak/
    prematch/
    postmatch/
  val/
    gameplay/
    timeout/
    setbreak/
    prematch/
    postmatch/
  metadata/
    windows.jsonl
    images.jsonl
    summary.json
    class_to_idx.json
```

Split by source video, not by frame or window, to avoid leakage.

Extraction protocol:

- Sample four-frame windows.
- Within each window, sample one frame per second for four seconds.
- For `gameplay`, start a window every 20 seconds.
- For non-gameplay classes, start a window every 1 second because these classes
  are much smaller.
- Resize extracted images to 512 pixels on the long edge.
- Use PIL Lanczos resampling and reasonable JPEG quality, currently quality 88.
- Store generated datasets under `/mnt/t/data/vball/skyball/gameplay/vN`, not in
  this repo.

Filename scheme:

```text
<match>_w<window_index>_t<start_ms>_s<frame_index>.jpg
```

For example:

```text
000016_hilarious_jerboa_w000123_t0001545562_s0.jpg
000016_hilarious_jerboa_w000123_t0001545562_s1.jpg
000016_hilarious_jerboa_w000123_t0001545562_s2.jpg
000016_hilarious_jerboa_w000123_t0001545562_s3.jpg
```

The shared prefix before `_s0`/`_s1`/`_s2`/`_s3` is the four-frame window ID
for temporal classifiers. Single-frame classifiers can train directly from the
ImageNet-style class folders.

Write JSONL manifests because they are streamable, easy to audit, and friendly
to PyTorch dataloaders. Store paths relative to the dataset root and include
enough timing metadata to reproduce the sample.

Each sample should include:

- `window_id` for four-frame windows, or `image_id` for single-image records
- `source_video`
- `raw_video_path`
- `frame_paths`
- `times_sec`
- `class_name`
- `binary_label`
- `region_start_sec`
- `region_end_sec`
- `boundary_margin_sec`
- `split`
- `sampling_seed`

Use image-level records for the single-frame baseline and four-frame window
records for temporal models. Do not duplicate image bytes in this repo.

## Label Policy

Treat active volleyball gameplay as the positive binary class. Treat labels such
as `set break`, `timeout`, `pre-match`, `post-match`, `warmup`, `set_break`,
`post_match`, and other non-active regions as negative classes. Preserve the
original class name in manifests so multiclass training can reuse the same data.

Class folder names:

- `gameplay`
- `timeout`
- `setbreak`
- `prematch`
- `postmatch`

Avoid ambiguous boundaries by dropping candidate windows that overlap exclusion
zones around labeled transitions:

- For the `pre-match` to `gameplay` interface, use a 20-second no-sample zone
  centered on the transition time.
- For all other labeled transitions, use a 5-second no-sample zone centered on
  the transition time.

## Engineering Notes

- Use `rg` for searches.
- Use structured parsers for Excel and manifests; avoid ad hoc string parsing.
- Keep generated review clips, checkpoints, and logs under `/mnt/t/output`, not in the repo.
- Keep extracted gameplay image datasets under `/mnt/t/data/vball/skyball/gameplay`.
- Use versioned output directories and do not overwrite completed runs.
- Use `conda run --no-capture-output -n pt5090new ...` for training and data jobs that need the SkyBall Python stack.
- Check `nvidia-smi` before launching GPU training.
- Use tmux for long-running training.

## Skills

- Use `$skyball-gameplay-dataset` for review clip generation and manifest dataset construction.
- Use `$skyball-gameplay-training` after the review clips and dataset have been approved.
