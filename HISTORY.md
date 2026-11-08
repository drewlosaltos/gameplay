# Gameplay history: superseded recipes and notes

Moved here on 2026-11-08 from `README.md`, `HANDOFF.md`, `model_registry.yaml` and
`workflows/gameplay_gate_validation.md`, unchanged apart from heading levels. None of this
describes the current production model. For that, see `REPRODUCE.md`. Paths and environment
names (`pt5090new`, `/mnt/t/data/vball/skyball/gameplay/...`) are as they were at the time.

Superseded `model_registry.yaml` entries (`previous_production_model`, `notes`,
`historical_production_model`, `prior_production_model`, `previous_qwen_v5_model`) are in
`docs/history/model_registry_history.yaml`.

## Indoor recipes before Qwen V8 (from README.md)

The old v5 launcher and two-stage recipes below are historical reproductions.

### Previous SigLIP recipe

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

The selected September 16 indoor model adapted on **v5**, then fine-tuned on
`human_app_v0`. An approved newer legacy dataset such as v6 can be passed with
`--legacy-data`; it does not require a version-specific training script.


## Gameplay gate: SigLIP-era baseline, commands and ONNX qualification (from workflows/gameplay_gate_validation.md)

The current baseline temporal policy uses a gameplay score threshold of `0.2`,
does not fill inactive gaps, and pads each retained interval by 30 seconds on
both sides. It was selected on these same ten matches, so this gate is a
non-regression contract, not an unbiased estimate of unseen-match transfer.

### Generate Evidence

Score all ten videos with the registered four-frame model. Use separate shards
when multiple GPUs are available:

```bash
conda run --no-capture-output -n pt5090new \
    python scripts/evaluate_gameplay_gate.py score \
    --output-root /mnt/t/output/skyball_gameplay/<study> \
    --num-shards 4 --shard-index <0-3> --device cuda:0
```

After every shard completes, generate gameplay coverage evidence:

```bash
conda run --no-capture-output -n pt5090new \
    python scripts/evaluate_gameplay_gate.py analyze \
    --output-root /mnt/t/output/skyball_gameplay/<study>
```

Generate the retrospective pipeline impact report with the quality-first
policy:

```bash
conda run --no-capture-output -n pt5090new \
    python scripts/evaluate_gameplay_pipeline_fp_impact.py \
    --study-root /mnt/t/output/skyball_gameplay/<study> \
    --pipeline-root /mnt/t/output/skyball_pipeline/<current-full-suite> \
    --output-dir /mnt/t/output/skyball_gameplay/<study>/<impact>
```

For the current SigLIP-512 baseline, pass `--threshold 0.2 --merge-gap-sec 0 --padding-sec 30`. Future checkpoints may select a different
policy, but selection must follow the safety-first rule above and record the
full searched grid as evidence.

### Apply the Gate

```bash
conda run --no-capture-output -n pt5090new \
    python scripts/validate_gameplay_gate.py \
    --report /mnt/t/output/skyball_gameplay/<study>/<impact>/pipeline_fp_impact.json \
    --output /mnt/t/output/skyball_gameplay/<study>/<impact>/validation_gate_result.json
```

Exit codes are:

- `0`: all six metrics pass and the population matches;
- `1`: one or more model metrics fail;
- `2`: evaluation membership or a fixed denominator changed.

Never relax or silently rewrite the existing contract for a changed evaluation
set. Add a new versioned contract and record its new population and baseline.


### Portable Runtime Qualification

Recommended integration candidates must also provide a portable ONNX graph and
real-window runtime parity evidence. Export the model with:

```bash
conda run --no-capture-output -n pt5090new \
    python scripts/export_gameplay_onnx.py \
    --checkpoint <checkpoint> \
    --output <bundle>/model.onnx \
    --device cuda:0
```

Validate ONNX against the source checkpoint with
`scripts/validate_gameplay_onnx.py`. Use `scripts/build_gameplay_trt.py` and
`scripts/validate_gameplay_trt.py` for local TensorRT build/execution evidence.
Publish the ONNX and evidence, not the machine-specific TensorRT engine.

The runtime input is normalized RGB `[batch, 4, 3, 512, 512]`; outputs are
two-class `logits` and the scalar `gameplay_score`. The classifier threshold
used for window-level F1 and the temporal gate threshold selected for safe
full-video filtering are separate operating points and must not be
interchanged.


### Production-frame event admission audit (September 16, 2026)

The user's August production run was audited directly at
`/mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1`.
Use `scripts/evaluate_gameplay_event_coverage.py` for this separate retrospective
contract (`gameplay_event_and_rally_frame_coverage_v1`), not the old July action-FP
population. Its requested metrics are:

1. Count confirmed GT events whose exact synchronized frame is excluded.
2. Percentage of the entire match admitted despite being outside every confirmed
   event's +/-3-second window. Union overlapping windows and clip to video bounds.
   Pool durations across matches, since source FPS differs.

The August run's GT hashes and exported event frames were verified, and every
saved production admission mask was reconstructed exactly. The candidate was
rescored on the identical primary videos (including the two 30-fps transcodes)
with the pipeline's unchanged OpenCV preprocessing, using PyTorch FP32. This is
not TensorRT runtime qualification or a detector/pipeline rerun.

| Model / frozen policy | Excluded GT events | Extra admitted time / total time |
|---|---:|---:|
| Saved production, threshold 0.20 / padding 30s | 0 / 8,848 | 52.488% |
| Two-stage human epoch 1, same production policy | 0 / 8,848 | 53.758% |
| Same new checkpoint, prior policy 0.15 / padding 10s | 0 / 8,848 | 46.425% |
| Production scores, same prior policy 0.15 / padding 10s | 13 / 8,848 | 44.766% |

The model alone admits 12.30 more minutes under the unchanged policy. The
previously selected new-model policy saves 58.72 minutes versus production,
including the policy change. No policy was tuned for this audit. About 59.63%
of production's extra admitted footage lies in same-set gaps of at most 60
seconds; 28.85% lies in longer same-set gaps, 7.40% between sets, and 3.94%
before/after the annotated match. These are event-derived categories, not
independently verified phase labels.

Per-match table, CSV, preprocessing provenance, four metric test results, and
full evidence:
`/mnt/t/output/skyball_gameplay/event_margin_full10_prod_vs_staged_20260916/report.md`.
The production registry, original suite, and existing July gate contract remain
unchanged.

The user-requested matched-policy follow-up shows why the shorter padding is
unsafe for the old production model: it excludes one LeftCoast–BayToBay event
and twelve USC–UCLA events. The new model excludes none at the same 0.15/10s
policy. Evidence: `production_with_prior_candidate_policy.json` in the audit
output directory.

## Gameplay dataset handoff, v0 extraction (from HANDOFF.md)

### Gameplay Dataset Handoff

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
