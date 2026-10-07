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
5. Canonical indoor training is **single-stage Qwen3.6 vision 672x384 FP8**.
   Use `scripts/train_qwen_canonical.py` and `config/indoor_canonical.json`:
   pretrained Qwen encoder, fresh binary head, no local gameplay initializer;
   four fully unfrozen epochs, four GPUs, effective batch 16, encoder LR 3e-6,
   head LR 1e-4. Use four frames, letterbox geometry, shared flips/5-degree
   rotation, light jitter, compiled FP8 training and FP32 validation.
   Select on clean val48 in `validation/gameplay_indoor_v8_val48_clean_v2.json`.
   No historical val20/val50 reevaluation is required. Read the temporal manifest,
   not ImageFolder: symlinked image directories contain excluded examples.
   Keep the two known reencoded training duplicates excluded. Future versions
   preserve these 48 validation sources and audit source-video aliases.
   Historical v5 Qwen and SigLIP launchers are reproduction recipes only.
   Both clean val48 classification and the full10 gate below are required.

For new gameplay versions with a static validation request, read
`workflows/new_dataset_static_val.md`. The comparison policy is to keep
validation source videos from the requested base dataset fixed and add new
matches to the training split unless the user asks to re-split validation.

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
- For legacy datasets, non-gameplay windows may start every 1 second. The v5
  policy is deliberately sparser and is documented in
  `workflows/gameplay_v5_dataset.md`: training uses a 10-second non-gameplay
  stride, frozen v2 validation retains its original 1-second non-gameplay
  stride, and expanded v5 validation uses a class-independent 10-second
  stride. Do not apply the training sparsification to frozen v2 validation.
- Resize extracted images to 1024 pixels on the long edge. Legacy v1/v2
  datasets extracted before June 28, 2026 may use 512 pixels on the long edge;
  new 512x512 training ablations should use 1024-long-edge extracted JPEGs.
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

## Gameplay Gate Validation

- Every completed gameplay training/fine-tuning candidate requires **two
  validation sources**: (1) held-out dataset classification metrics, including
  F1 on clean val48 for canonical indoor runs; (2) the canonical ten full
  evaluation matches. Do not declare model validation complete from F1 alone.
- For the second source, report per match and pooled: **number of confirmed GT
  events excluded** by the final gameplay admission ranges, and **extra footage
  admitted as a percentage of the entire match**. Extra footage means frames
  farther than +/-3 seconds from every confirmed synchronized GT event. Union
  overlapping event windows, use exact source-frame coordinates, and duration-
  weight the pooled percentage. Outside-event footage is a proxy, not manually
  verified non-gameplay. Prioritize zero excluded events over filtering more.
- Bind these evaluations to the canonical primary-video root
  `/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1`,
  including its two 30-fps transcodes and remapped GT. Use production frame
  sampling/preprocessing and verify cached input/checkpoint identities before
  reuse. `scripts/evaluate_gameplay_event_coverage.py` implements the metrics.
- State threshold, padding per side, merge gap, and runtime for every result.
  Compare candidate and production at identical settings, then separately show
  their selected settings. Include both validation sources and evidence paths
  in the result ledger/model registry. These ten matches are a second source
  of validation, not an independent test set after model/policy selection.
- Use `workflows/gameplay_gate_validation.md` for full-video gameplay-gating
  studies against the ten SkyBall evaluation matches.
- The canonical joint safety/effectiveness non-regression contract is
  `validation/gameplay_gate_eval10_v2.json`; apply it with
  `scripts/validate_gameplay_gate.py` to the report produced by
  `scripts/evaluate_gameplay_pipeline_fp_impact.py`.
- Treat full-video gate safety and target-FP rejection as the primary gameplay
  R&D criteria. First require 100% annotated-gameplay, confirmed-action, and
  full-rally retention plus zero rejected matched actions or valid plays. Among
  policies meeting every safety constraint, maximize rejection of target
  false positives. Report combined-validation gameplay F1 alongside the gate;
  classifier F1 alone cannot qualify or recommend a model.
- The target false-positive definition is an unmatched predicted action more
  than three seconds from every confirmed synchronized GT action. Do not use
  all unmatched predictions as the non-gameplay FP denominator.
- Treat changed match membership, GT action/rally counts, target-FP count, or
  matched-action/valid-play count as a new validation population. Create a new
  versioned contract instead of weakening the existing gate.
- Every recommended model entry in `model_registry.yaml` must record gameplay
  F1, the selected temporal policy, all safety recalls/counts, target FP
  current/rejected/remaining counts and fraction, and evidence paths.
- This gate is currently retrospective and standalone. It does not authorize
  gameplay integration or changes in the production pipeline.

## Skills

- Use `$skyball-gameplay-dataset` for review clip generation and manifest dataset construction.
- Use `$skyball-gameplay-training` after the review clips and dataset have been approved.
