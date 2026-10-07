## Production promotion — 2026-10-07

The user approved the V8 single-stage Qwen candidate as production, the exact
FP8 recipe as the indoor default, and clean val48 as default validation (no
val20/val50 reevaluation). See `workflows/qwen_v8_production_20261007.md`,
`config/indoor_canonical.json`, and `scripts/train_qwen_canonical.py`.
Model: `v8_val48_qwen672x384_single_stage_fp8_20261006_r3` (epoch 3).
Pipeline release: `skyball-prod-20261007-r1`; threshold 0.9, pad 5s/side, gap 0.
Publication/evidence: `/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1`.
Prior entries below are historical.

# Gameplay Dataset Handoff

Status as of restart request:

- `AGENTS.md` has been updated with the corrected raw-video protocol.
- `gameplay.xlsx` timestamps refer to full raw match videos under
  `/mnt/t/data/vball/skyball/raw_video`, not `unified_dataset/images_full`.
- All 36 workbook raw videos were resolved on `googdrive:skyball/raw_video` and
  copied locally under:
  `/mnt/t/data/vball/skyball/raw_video/indoor_endline/youth/`
- Duration check passed for all 36 local raw videos against workbook max
  `end_sec` with no mismatches over 0.25s.
- Raw sync metadata is in:
  `/mnt/t/data/vball/skyball/gameplay/v0_raw_sync/`
- The current `/mnt/t/data/vball/skyball/gameplay/v0` is incomplete and should
  be rebuilt with `--force`.

Important script:

```bash
scripts/build_gameplay_v0_dataset.py
```

The script has two commands:

```bash
conda run --no-capture-output -n pt5090new \
  python scripts/build_gameplay_v0_dataset.py sync-raw

conda run --no-capture-output -n pt5090new \
  python scripts/build_gameplay_v0_dataset.py extract --force
```

Do not rerun `sync-raw` unless needed; the raw videos are already local.

Extraction protocol to preserve:

- Output root: `/mnt/t/data/vball/skyball/gameplay/v0`
- Split by source video into `trn/` and `val/`.
- Class folders: `gameplay`, `timeout`, `setbreak`, `prematch`, `postmatch`.
- Four-frame windows, one frame per second.
- `gameplay` window stride: 20 seconds.
- non-gameplay window stride: 1 second.
- no-sample zone for `pre-match` <-> `gameplay`: 20 seconds centered on transition.
- no-sample zone for all other transitions: 5 seconds centered on transition.
- Image size: 512 pixels on long edge.
- Filename scheme:
  `<match>_w<window_index>_t<start_ms>_s<frame_index>.jpg`

Performance notes:

- The original OpenCV per-timestamp seek extractor was stopped because it became
  extremely slow around `000019_monstrous_gibbon`, writing about one frame every
  8-12 seconds.
- A streaming `ffmpegcv` decode path was also tested and was slower for this
  workload because it decoded full videos and resized in Python.
- The current script has been patched to use an ffmpeg 2-fps temporary cache per
  video, with Lanczos scaling and JPEG writing in ffmpeg, then hardlinks cache
  frames into the requested ImageNet-style filenames.
- Smoke test for this ffmpeg-cache path passed, including hardlinking duplicate
  timestamp outputs.
- Next step is to run the full extraction again:

```bash
conda run --no-capture-output -n pt5090new \
  python scripts/build_gameplay_v0_dataset.py extract --force
```

After extraction completes, validate:

```bash
find /mnt/t/data/vball/skyball/gameplay/v0 -type f -name '*.jpg' | wc -l
cat /mnt/t/data/vball/skyball/gameplay/v0/metadata/summary.json
```

Expected full logical image-file count is about `66,980` window-frame paths,
with hardlinks reducing duplicate storage for overlapping negative windows.
