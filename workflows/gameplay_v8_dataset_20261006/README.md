# V8 dataset build, 2026-10-06: one-off scripts as they ran

These are the driver, audit and overlay scripts that built
`v8_val48_trainclean_overlay_20261006_r4`, copied unchanged from
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/` and
`/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/`. They are a record,
not tools: their paths are hardcoded, several of those paths have since moved,
and each refuses to run once its output exists. How they fit together, and
which steps the reusable scripts in `scripts/` cover, is in
`workflows/gameplay_v8_dataset.md`.

| Script | Stage |
|---|---|
| `run_missing_video_sync.sh`, `resume_missing_video_copy_20261006.sh`, `run_final_check_after_copy.sh` | Copy missing raw videos from the shared Drive folder (`rclone copy`, missing files only) and check them |
| `check_v8_raw_composite.py` | Checksum the selected raw videos across the main and recovered roots |
| `run_review_clips_balanced.sh` | Review clips shown to the user before extraction |
| `extract_available_videos.py` | Run `scripts/extract_indoor_v8_video.py` per video, 8 at a time, as raw copies finished |
| `finalize_when_extracted.py` | Run `scripts/finalize_indoor_v8_dataset.py` once every video check passed |
| `audit_early_raw_reads.py` | Find extractions that started before rclone finished the raw file (11 sources) |
| `build_clean_v8_r2.py` | Make r2: hardlink r1's verified frames, drop the 11 suspect sources for fresh extraction, point 8 at recovered raw files |
| `audit_reextracted_v8_images_when_ready.py` | Fully decode every re-extracted r2 JPEG from those 11 sources |
| `check_v8_raw_byte_duplicates.py`, `screen_v8_visual_duplicates.py` | Look for the same match in train and validation (byte-identical, then re-encoded) |
| `prepare_v8_trainclean_overlay.py` | r3: drop the 2 training re-encodes (val50; not used by prod) |
| `prepare_v8_clean_val48_overlay.py` | r4: also drop their 2 validation twins (val48; prod). Reusable version: `scripts/build_indoor_v8_clean_overlay.py` |
