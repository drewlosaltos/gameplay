# Gameplay Handoff

**Current v8 work (October 6, 2026):** The live single-pane workboard is
`tmux attach -t gameplay_v8_dashboard_20261006`. Its controller and status files
are under `/mnt/t/output/skyball_gameplay/v8_workflow_20261006`. The workboard
tracks the Google Drive copy, v8 extraction, FP8 pilots, Qwen stage-two run,
single-stage challenger, and ten-match gate. Read the status JSON files and
live tmux sessions there before restarting a job; completed outputs use versioned
paths and should be preserved.

At 22:04 PDT, the corrected stage-two run has completed epoch 3 and is
training epoch 4 (batch 101/29,856). Clean val48 F1 by epoch is 0.956685,
0.959915, and 0.959542; epoch 2 remains selected. All four GPUs are active.
At the observed epoch duration (~93 minutes), stage-two training is projected
to finish around 23:40 PDT, followed by same-population classification and the
common ten-match gate. The single-stage challenger is queued until the pilot
artifacts and stage-two common-policy gate are complete; the full comparison
is currently projected for October 7 morning. Live status is in the workboard
and JSON files under `/mnt/t/output/skyball_gameplay/v8_workflow_20261006`.

The four-GPU four-epoch batch-16 FP8 pilot completed with best F1 0.9003 on the
fixed 640-window pilot validation, versus 0.9026 for batch 8. Its steady epoch-2
throughput was 21.2% above batch-8 FP8, so it passed the pre-set four-epoch
F1 and speed rule. The full-v8 single-stage challenger will use batch 16;
evidence is `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/batch16_four_epoch_result.json`.
The stage-two baseline remains batch 8 for recipe parity.
**Validation split integrity:** A raw-frame audit
confirmed that two frozen val20 videos are re-encodes of training videos:
`vid_mtyozrerb65868` ↔ `vid_mtxwk77pba622d` (the training copy was already in
the historical production dataset) and `vid_mtzaxwm0617fe8` ↔
`vid_mu31qed1aaf075` (added with v8). They account for 485/4,828 val20
windows. Therefore historical val20 and combined val50 F1 are contaminated
continuity figures, including the exploratory v8 stage-two r1 epoch-1 F1
0.9633867 on val20 and epoch-2 F1 0.9580585 on val50. The r1 run was stopped
and explicitly invalidated for model selection; a first train-only correction
r2 was also stopped before epoch 1 when we recognized that online selection
must exclude the validation copies too. Evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_validation_leak_audit.json`.

A follow-up one-frame-per-second motion screen of six ambiguous same-court
cross-split pairs found maximum aligned residual correlation 0.175, versus
1.000 for a confirmed re-encode. It found no additional aligned recording
among those six; videos from the same venue or disjoint segments of one match
remain possible. Evidence and the exact method are in
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/dedup_motion_probe_r1/result.json`.

The versioned selection contract is
[`validation/gameplay_indoor_v8_val48_clean_v2.json`](validation/gameplay_indoor_v8_val48_clean_v2.json).
It records both excluded train sources, both excluded frozen validation sources,
all 48 selected validation IDs, exact window counts, and dataset hashes. The
final comparison watcher verifies this contract against its scored cohorts.

The clean manifest-only temporal overlay is
`/mnt/t/data/vball/skyball/gameplay/v8_val48_trainclean_overlay_20261006_r4`.
It excludes both training copies and both validation copies: 1,029 train
sources/238,845 windows and 48 validation sources/16,281 windows (18 frozen
plus 30 new). The train and val image folders are symlinks to the verified r2
JPEGs; use its filtered manifests through the temporal dataset loader, not
ImageFolder. The ready marker and counts are in
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/val48_trainclean_dataset_ready.json`.
The corrected four-GPU FP8 Qwen stage-two run is
`/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_stage2_4gpu_20261006_r3`.
It uses the historical production initializer, four high-resolution epochs,
batch 8, and selects checkpoints on clean val48 F1. Its live configuration,
manifest hashes, 1,029/48 source split, excluded source IDs, initializer hash,
and nine code-snapshot hashes passed an independent check recorded in
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/stage2_clean_run_integrity_preflight_r3.json`. Its launcher, validation,
full10 gate, policy search, TensorBoard mirror/server, single-stage successor,
and final comparison watcher run in tmux; see their status JSON files in the
work directory. TensorBoard ports are 6008 for stage two and 6009 for the
single-stage successor. The original frozen val20 and full val50 are evaluated
separately from the published r2 dataset for historical continuity, explicitly
labeled contaminated; clean val18/new30/val48 and the canonical full10 gate
are the comparison sources. The full10 gate is the primary model decision
criterion. Saved val20 predictions provide a clean historical val18 reference:
production epoch 1 F1 0.960524, production final epoch 4 F1 0.965517, and the
invalidated exploratory v8 epoch 1 F1 0.962736 on the same 4,343 windows.
See `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/epoch1_clean_val18_comparison_r1.json`.
The corrected run scored clean val48 F1 0.956685 after epoch 1 and
0.959915 after epoch 2 (16,281 windows); epoch 2 improved by 0.323
percentage points and is the selected checkpoint. Its same-population
production comparison is queued after training; do not compare these val48
scores directly with the historical production val18 figure above. The
per-epoch evidence is
`/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_stage2_4gpu_20261006_r3/epoch_metrics.jsonl`.
Model-reviewed label provenance remains a separate caveat.
The full-v8 single-stage launcher passed a preflight against the clean
255,126-window overlay and versioned contract: the four-epoch pilot qualifies,
FP8/default compile and global batch 16 are selected, the pretrained Qwen
snapshot and TorchAO overlay exist, and the versioned output path is unused.
See `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/single_stage_full_v8_preflight_r3.json`.

The queued single-stage run has its own TensorBoard sidecar and server watcher;
after its training starts, they will mirror each clean val48 epoch on port 6009.
They wait without creating the versioned training directory early. Its launcher
now takes the GPUs after stage-two classification, full10 scoring, and the
common-policy gate finish; the CPU-only stage-two policy search can finish in
parallel with single-stage training.
The ten-match policy search was profiled on cached production scores: three
representative policies implied roughly 51 seconds for the 272-policy grid,
so that CPU search is not expected to delay the next training handoff. See
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/policy_search_runtime_profile_r1.json`.
The gate, fixed-population contract, full10 input, and event-coverage tests
passed together (18 tests) in `pt5090new` on October 6.
The queued final comparison report includes separate legacy-v5, frozen-val20,
new30, combined-val50, clean-val18, and clean-val48 F1 columns alongside common and selected full10 gate
metrics, pooled and per-match inference runtime; its watcher waits in
`gameplay_v8_final_comparison_20261006`. The
report now records a research recommendation only if a v8 candidate passes
every selected-policy safety check and rejects more target false positives
than production; a val50 F1 gain alone does not qualify it. The updated
watcher runs from `final_comparison_val48_r6.log` and makes no
production pipeline change.
After the report finishes, review its selected-policy safety and target-FP
results against production. If a v8 candidate qualifies for recommendation,
record its clean validation F1, full10 gate counts, temporal policy, and
evidence paths in `model_registry.yaml` before treating it as recommended.
An optional batch-20 FP8 pilot was stopped after two epochs because steady
throughput improved only 0.75% over batch 16, below the preset 5% continuation
rule; its best pilot F1 was 0.8243 versus 0.9003 for batch 16. Evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/batch20_two_epoch_speed_result.json`.
Batch 16 with FP8 `default` compile mode remains selected. A two-epoch
batch-16 `reduce-overhead` compile pilot reached F1 0.8986, comparable with
default mode, but its steady throughput gain was only 1.32% versus the
same-horizon default pilot (2.52% versus the separate four-epoch default run).
Both are below the pre-set 5% gain needed to switch modes. Evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/reduce_overhead_fp8_same_horizon_comparison.json`.
The single-stage launcher reads the decision and keeps `default`.

The frozen val20 plus new30 v8 split is
[`validation/gameplay_indoor_v8_val50_v1.json`](validation/gameplay_indoor_v8_val50_v1.json).
Its 20 sources, 4,828 windows, and base manifest SHA match
`human_app_v5_val20_20260925` exactly; evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/val20_lineage_check.json`.
An independent audit of the published v8 manifest also found the frozen
val20 JSONL lines byte-for-byte identical, in the same order, with the same
20 sources and class counts; see
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_published_val20_exact_audit.json`.
The frozen split is not entirely human-labeled according to its own metadata:
5 sources (1,420 windows) say `human_labeled`, while 15 sources (3,408 windows)
say `model_reviewed_pending_human_acceptance`. All 30 new validation sources
also carry pending model-reviewed labels. Report val20, new30, and val50 F1 as
provisional; provenance evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_val20_provenance_audit.json`.
The planned training manifest also marks 230,155 of 239,338 windows as
`model_reviewed_pending_human_acceptance`; see
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_cohort_counts_preflight.json`.
Val20 has 4,828 windows (62.7% gameplay); new30 has 11,938 (76.4%
gameplay), so the expanded val50 window-level F1 is weighted toward new30.
Report the cohorts separately.
The v8 cohort summarizer was rehearsed against 16,766 synthetic predictions
derived from the published validation manifest: it resolved all 20 frozen and
30 new source videos, matched cohort window counts, and balanced its score
histograms by true class. These synthetic 1.0 metrics are plumbing checks,
not model quality; see
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/cohort_summarizer_smoke_r1/README.txt`.
The full-v8 dataset was published at
`/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2` after extraction
and a composite checksum audit passed for all 1,081 selected raw videos with
zero mismatches or missing files. Its manifest contains 256,104 windows and
1,024,416 images across 1,031 train, 20 frozen val20, and 30 new30 sources;
the recorded manifest SHA-256 values were independently rechecked. Evidence is
`/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/composite_check_r1/result.json`,
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/dataset_ready.json`, and
the dataset's `metadata/summary.json`. The first Qwen stage-two run started at
13:03 PDT on all four GPUs with FP8, the production legacy initializer, 384x672
input, four epochs, and effective batch 8. Its run directory was
`/mnt/t/output/trn/skyball_gameplay/v8_val50_qwen672x384_stage2_4gpu_20261006_r1`;
it was stopped and invalidated after the re-encode leak was discovered.
An independent raw-file leakage preflight found 1,081 distinct paths and
1,081 distinct byte sizes across the 1,031/20/30 train/val20/new30 split,
excluding exact-byte duplicate videos across cohorts. It does not rule out
re-encodes of the same visual content. Evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/raw_byte_duplicate_preflight.json`.
An independent run-integrity preflight rehashed the dataset manifest, dataset
summary, initializer, and all nine snapshotted source files, and verified the
raw checksum gate and four-GPU FP8/batch-8 configuration; every check passed.
See `/mnt/t/output/skyball_gameplay/v8_workflow_20261006/stage2_run_integrity_preflight.json`.
A recipe parity preflight
against the production Qwen run passed 34 checks, including pixel-identical
letterbox and rotation outputs on representative images; see
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/stage2_recipe_parity_preflight.json`.
A separate four-epoch
single-stage pilot reached F1 0.9026 on fixed val8 and passed its pre-set
qualification rule; the full-v8 challenger waits for the stage-two gate. New30
labels are model-reviewed and provisional. The canonical ten-match gate remains
retrospective validation and does not authorize pipeline promotion.
The production full10 score cache was revalidated against all ten canonical
primary-video manifests, checkpoint SHA-256, score counts, synchronized-GT
hashes, and a fresh one-match rescore (maximum score difference 1.49e-7).
The primary-video audit also confirmed exactly two 30-fps transcodes and
checked every remapped original/synchronized frame coordinate and preserved
label across their 501 and 450 GT rows. Evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/prod_full10_cache_verified.json`.
The queued single-stage comparison reuses the stage-two run's production
val20/new30/val50 and legacy-v5 classification results after checking the
production checkpoint SHA-256 and v8 manifest SHA-256. Each validation job
scores its candidate checkpoint on the canonical ten matches.
The reuse audit will be written to
`/mnt/t/output/skyball_gameplay/v8_single_stage_val48_trainclean_validation_20261006_r3/production_classification_reuse.json`.
The pilot's saved FP8 Qwen checkpoint also passed a strict CPU reload through
`load_checkpoint_model` (339 state keys, four frames, 384x672 input), so the
queued evaluators have a verified checkpoint format to consume.
A two-window CPU evaluation smoke then passed the full saved-checkpoint load,
preprocessing, forward, metrics, and prediction-writing path without taking a
training GPU. It is a compatibility check, not a quality estimate; see
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/fp8_eval_smoke_v1/result.json`.

The 983.8-GiB raw-video copy and all 1,061 new-source extractions completed on
October 6. All 1,061 video checks passed; 515,920 fresh r2 JPEGs were written.
Five planned audio-tail windows were dropped, accounting for the 20-image
difference from the pre-extraction estimate. The composite checksum audit and
dataset finalization have passed. The final image accounting is 508,496 verified
reused JPEGs plus 515,920 fresh JPEGs; evidence is
`/mnt/t/output/skyball_gameplay/v8_workflow_20261006/v8_image_accounting_preflight.json`.
The manifest finalizer now validates JPEG presence and nonzero size in bounded
128-window batches with 16 workers. On a 4,000-image NFS sample, checks rose
from roughly 350 files/s sequentially to roughly 5,000 files/s; a 2,003-path
comparison including missing, empty, and directory paths matched the prior
predicate exactly. The finalizer still checks every image before publication.

**Previous handoff (September 24, 2026): [Best recent models and reboot notes](workflows/recent_best_models_20260924.md).**

The dataset notes below are historical. Do not use their old extraction commands as current reboot/resume instructions.

## Historical gameplay dataset handoff

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
