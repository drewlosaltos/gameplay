# Gameplay v8 dataset: how prod's training data was built

Prod gameplay (`v8_val48_qwen672x384_single_stage_fp8_20261006_r3`) trained on
`derived/gameplay/v8_val48_trainclean_overlay_20261006_r4` under
`SKYBALL_DATA_ROOT` (default `/mnt/t/data/vball/skyball/unified_dataset`). This
is how that folder was made on 2026-10-06, what can be re-run, and what to do
for a new version. The one-off drivers and audits are kept unchanged in
`workflows/gameplay_v8_dataset_20261006/`.

## Inputs

| Input | Where | Identity |
|---|---|---|
| V8 label release (`manifest.json`, `phases.csv`, `videos.txt`, …) | `annotations/gameplay/indoor/skyball-gameplay-finetune/v8` (Drive master: `googdrive:skyball/datasets/annotations/gameplay/indoor/skyball-gameplay-finetune/v8`) | sha256 `dd355b2d…`, `59a6b22a…`, `66af1e95…` (recorded in the dataset summary) |
| v5 dataset: the frozen val20 and frames to reuse | `/mnt/t/data/vball/skyball/gameplay/human_app_v5_val20_20260925` (outside unified_dataset) | `windows.jsonl` sha256 `c90af50c…` |
| Raw match videos | `/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user`, plus 8 re-downloaded files in `…/v8_recovered_20261006_r1` | checked against the shared Drive folder with `rclone check --checksum` |
| Shared Drive listing (file sizes and times) | `/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/shared_video_listing.json` | |
| Train/validation leak audit | `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_validation_leak_audit.json` | 2 confirmed re-encoded pairs |

## Steps

1. **Split** (`scripts/plan_indoor_v8_split.py`, seed 20261006). Keeps v5's 20
   validation videos (val20) and picks 30 videos new in v8 (22 `vid_`, 8
   `original_`), stratified by duration and gameplay share. The other 1,031
   videos are training. Output: `validation/gameplay_indoor_v8_val50_v1.json`.
2. **Window plan** (`scripts/prepare_indoor_v8_dataset.py`). Turns label
   segments into four-frame windows using the same rules as
   `build_gameplay_v0_dataset.py`: 4 s clips; boundaries excluded 20 s around
   prematch/gameplay changes and 5 s around others; gameplay stride 20 s in
   train and 10 s in new30, other classes every 10 s; activity-category
   overrides. Output: work
   folder `/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1`, with
   251,281 new windows in `planned_windows.jsonl` and `windows_by_video/`.
3. **Reuse v5 frames** (`scripts/reuse_indoor_v8_images.py`). Hardlinks
   frames v5 already decoded at the same video and half-second, and puts val20
   in place unchanged. 482,760 training frames were reused.
4. **Extract the rest** (`scripts/extract_indoor_v8_video.py`, one call per
   video, driven by `extract_available_videos.py`). Seeks with ffmpeg (CUDA
   decode), resizes with PIL Lanczos to 1024 px on the long edge, and saves
   JPEG quality 88. Review clips (`run_review_clips_balanced.sh`) went to the
   user before full extraction.
5. **Finalize** (`scripts/finalize_indoor_v8_dataset.py`, via
   `finalize_when_extracted.py`). Validates every video's frames and writes
   `metadata/` (`windows.jsonl`, `images.jsonl`, `summary.json`). This made r1,
   `v8_val50_20261006`.
6. **r2, clean re-extraction.** `audit_early_raw_reads.py` found 11 videos
   whose extraction began before rclone finished copying the raw file.
   `build_clean_v8_r2.py` hardlinked r1's other 508,496 frames into r2 and
   pointed 8 videos at re-downloaded raw files. Steps 4–5 then re-ran for the
   11, and `audit_reextracted_v8_images_when_ready.py` decoded every new JPEG.
   Result: `v8_val50_20261006_r2`, 256,104 windows and 1,024,416 frames. Its
   metadata is kept in r4's `provenance/v8_val50_metadata/`.
7. **Leak audit.** `check_v8_raw_byte_duplicates.py` and
   `screen_v8_visual_duplicates.py` looked for the same match in train and
   validation. Two training videos turned out to be re-encodes of two val20
   videos (485 val20 windows, 10%). The audit JSON names the pairs.
8. **Clean view** (`scripts/build_indoor_v8_clean_overlay.py`). Starts from
   r2's metadata and drops both re-encoded pairs: the training copies from
   train, and their validation twins from val. Result: r4, with 255,126
   windows, 1,029 training videos and 48 validation videos (clean val48: val18
   plus new30). The frames are r2's; folder scans also see the 3,912 frames of
   the 4 dropped videos, so train from `metadata/windows.jsonl` only.

## Re-run checks (2026-11-06)

- Step 1 reproduces `validation/gameplay_indoor_v8_val50_v1.json` byte for
  byte (sha256 `aca12f9c…`), reading labels from `annotations/gameplay`.
- Step 2 reproduces r1's `planned_windows.jsonl` (`2c2295b8…`) and every
  per-video file.
- Step 8 on `provenance/v8_val50_metadata` reproduces r4's `windows.jsonl`
  (`8b154d8d…`, the identity `scripts/train_qwen_canonical.py` checks), plus
  `images.jsonl`, `split.json` and `class_to_idx.json`.
- Steps 3–6 were not re-run. They decode video, so the frames depend on the
  ffmpeg/GPU decoder and on v5's earlier frames, and would not be byte-identical.
  The frames in `derived/gameplay` are the record.

## Known gaps

- No script wrote `v8_validation_leak_audit.json` or the probes it cites
  (`visual_duplicate_probe.json`, `visual_duplicate_offset_probe.json`,
  `cross_split_same_duration_probe.json`). They were assembled during the
  session from the two screens above. The pairs are also listed in
  `validation/gameplay_indoor_v8_val48_clean_v2.json`
  (`excluded_train_reencodes`, `excluded_val20_reencodes`).
- The v5 base dataset and raw videos live outside `unified_dataset`.
- The 30 new validation videos have model-reviewed labels awaiting human
  acceptance, so val48 F1 is provisional.

## Building a new version

1. Put the new label release under `annotations/gameplay/indoor/…/vN` (Drive
   first).
2. Write a new split with `plan_indoor_v8_split.py --v8 … --base … --output
   validation/gameplay_indoor_vN_….json`. Freeze the previous validation set;
   don't redraw it.
3. Run `prepare_indoor_v8_dataset.py` with new `--work` and `--dataset`
   folders, then the reuse, extract and finalize steps. Make sure each raw file
   has finished copying before extracting it: check it against the Drive
   listing, which is the step-6 lesson.
4. Screen for train/validation re-encodes. If any are confirmed, run
   `build_indoor_v8_clean_overlay.py --base-metadata <base>/metadata --output
   <new root>`, then hardlink the base `train/` and `val/` folders into it (no
   symlinks in `unified_dataset`).
5. Put the result under `derived/gameplay/<name>_<date>_r<N>`, give it a new
   validation contract, and point `config/indoor_canonical.json` at it only
   when promoting.
