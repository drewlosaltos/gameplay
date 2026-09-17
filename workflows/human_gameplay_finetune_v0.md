# Human gameplay v0 fine-tuning

This dataset is separate from the older model-assisted gameplay labels. The
user requested initialization from the existing indoor gameplay model and
fine-tuning exclusively on the human-labeled training matches.

Legacy provenance note: the new annotation README calls the older corpus
"model-assisted segmentation." June 28 user history confirms workbook updates
and manual review of sample clips, including correction of a raw-video mapping
bug, but does not establish who edited individual label intervals. The 123
videos shared by the local v2 and v5 workbooks have identical category/set/time
sequences. This does not rule out manual edits before v2 or outside the recorded
workflow; do not describe the legacy labels as verified untouched LLM output.

## Inputs and split

- Drive annotations: `googdrive:skyball/datasets/gameplay/indoor/skyball-gameplay-finetune/v0`
- Raw videos: `googdrive:skyball/raw_video/indoor_endline/skyball-user/<video_id>.mp4`
- Local annotation snapshot: `/mnt/t/data/vball/skyball/gameplay/ground_truth/human_app_v0`
- Dataset: `/mnt/t/data/vball/skyball/gameplay/human_app_v0`
- Build evidence and review clips: `/mnt/t/output/skyball_gameplay/human_app_v0_20260916`
- Split contract: `validation/gameplay_human_app_v0_split_v1.json`

The revised CSV has the legacy nine-column schema followed by app provenance.
Use `video` and `category`, with precise `start_sec` / `end_sec`; `label` is a
match description. Preserve the original CSV. The `set` column is derived,
and the `note` column is empty. Do not infer missing opening labels.

The user explicitly selected these five validation videos; all other 45 are
training videos:

- `vid_mu4jn79xb2b1e8`
- `vid_mu3nfyd5cfea51`
- `vid_mu1itmwmee3ae4`
- `vid_mtq23tfr7323cb`
- `vid_mtpt8nn9bc1bbf`

No legacy examples enter either split. The completed dataset contains 11,309 four-frame
windows / 45,236 JPEGs: 39,556 training images and 5,680 validation images.
All 50 raw files passed checksum and duration checks. All JPEGs and dataset
manifests passed validation, with no source split leakage.

## Extraction

Both splits use gameplay start stride 20 seconds and negative start stride
10 seconds. Windows contain four frames one second apart. Transition zones
are 20 seconds centered on pre-match/gameplay transitions and 5 seconds at
other transitions. Images have a 1024-pixel long edge, PIL Lanczos resizing,
and JPEG quality 88. `scripts/extract_gameplay_pil.py` uses the existing
sampler and its nearest-half-second decoded-frame convention, streaming only
selected frames rather than writing a full-video temporary cache.

Each raw file must match the Drive MD5 and pass duration checks before it is
used. Copy only reconciled missing files, preserving existing data. Render
twelve four-second review samples per selected source with class/time overlays.

The build controller is `prepare_extract.py` in the evidence directory. Its
`video_checks/` directory records per-video hashes, durations, and image counts.
The final `metadata/human_provenance.json` links the source snapshot and split.
The controller and download commands run in dedicated tmux sessions.

## Evaluation and training

Baseline / initializer: the registry's selected indoor SigLIP-Base 512 model,
`v5_siglip_base_f4_avg_h512w512_step4s2_lr1e4_blr5e6_20260719`.

1. Validate exact source membership, timing, manifests, and all JPEGs.
2. Evaluate the existing checkpoint on the five selected validation matches.
3. After the user's visual review, fine-tune that same checkpoint on the 45
   human-labeled training videos. Training is not authorized to begin merely
   because extraction or baseline evaluation completed.
4. Compare the old and fine-tuned models on the exact same validation manifest,
   including fixed-threshold F1, precision, recall, accuracy, ROC-AUC, confusion,
   and per-source metrics. Record any threshold-tuned metrics separately.

The initial fine-tune uses `scripts/train_siglip_temporal.sh` with
`INIT_CHECKPOINT` set to the baseline, 512x512 transforms, four epochs, AdamW,
head LR 1e-4 and backbone LR 5e-6, step decay after epoch two, and effective
batch size 128. Recheck GPU availability before launching. Preserve completed
runs under a new versioned output directory.

The existing full-video gate remains required before recommending a production
replacement. This work does not itself authorize production integration.

## Completed results — September 16, 2026

The user approved all four review videos. Four fine-tuning epochs completed;
the existing launcher's validation-accuracy selection retained epoch one.
Both models were evaluated on identical 1,420-window validation populations
at the fixed 0.5 threshold.

| Metric | Old model | Fine-tuned model |
|---|---:|---:|
| Gameplay F1 | 91.3862% | 91.7688% |
| Accuracy | 89.3662% | 90.0704% |
| Precision | 95.3571% | 98.2500% |
| Recall | 87.7327% | 86.0898% |
| ROC-AUC | 0.972769 | 0.982478 |

The modest F1 gain is 0.3826 percentage points. False positives fell from 39
to 14, while false negatives increased from 112 to 127. Later epochs had
validation accuracies of 81.690%, 82.042%, and 83.944%, below epoch one's
90.141% training-time validation accuracy. The standalone evaluation above
is the canonical comparison for the saved checkpoint.

- Result ledger: `validation/gameplay_human_app_v0_finetune_v1.json`
- Comparison: `/mnt/t/output/skyball_gameplay/human_app_v0_20260916/comparison.md`
- Review page: `/mnt/t/output/skyball_gameplay/human_app_v0_20260916/reviews/index.html`
- Run: `/mnt/t/output/trn/skyball_gameplay/human_app_v0_siglip512_prod_finetune_20260916`
- Best checkpoint: `best_siglip_base_f4_avg_binary_h512w512_step4s2_pre_drop010.pth.tar`

## Failure and variety audit — September 16, 2026

The follow-up diagnostic joined all validation predictions to source timing,
visually inspected 28 selected four-frame errors, inspected temporal contact
sheets from five longer source excerpts, and reviewed one gameplay view per
training match. No additional training or label changes were made.

Two matches (`vid_mtq23tfr7323cb` and `vid_mu1itmwmee3ae4`) account for 114 of
141 fine-tuned-model errors. Observed failures include clear serve preparation
in a wide view with distant players and huddles/resets inside gameplay regions.
The training set spans 29 program IDs but contains repeated-looking tournament
views; program counts do not measure unique venues. Targeted camera/scene and
within-set behavior variety is a stronger data hypothesis than simply adding
more nearby frames. Its benefit remains untested.

Two timeout intervals in `vid_mu1itmwmee3ae4` (2864.61–2939.66s and
3056.00–3123.88s) contain apparent rally action in the inspected context.
Audit these against the footage and label policy before treating every false
positive as model failure. Preserve the original labels and comparison; any
agreed correction requires a versioned benchmark and re-evaluation of both models.

Evidence directory:
`/mnt/t/output/skyball_gameplay/human_app_v0_failure_audit_20260916`
contains `index.html`, `report.md`, per-match error sheets, context MP4s,
the training-view overview, and reproducible joined prediction diagnostics.

## Full-court experiments — September 16, 2026

The user authorized independent improvement experiments after the failure audit.
All three started from the same registered v5 checkpoint, trained exclusively
on the 45 human training matches, and preserved the five-match validation set
and its original labels. Use the unchanged 0.5 threshold for headline metrics.

New trainer options support full-frame geometry, F1/balanced-accuracy checkpoint
selection, optional positive-class loss weighting, and FP32 validation during
AMP training. Ten targeted trainer/geometry/DDP regression tests passed.

| Experiment | Human validation F1 | Recall | Precision |
|---|---:|---:|---:|
| Gentler LR, previous augmentation | 90.515% | 84.666% | 97.233% |
| Gentler LR, full court and light color augmentation | 95.380% | 94.962% | 95.801% |
| Same full-court recipe, gameplay loss weight 1.5 | 95.264% | 95.838% | 94.697% |

Each run used head LR 2e-5, encoder LR 1e-6, four epochs, step decay after two,
effective batch 128, and seed 20260916. Full-court runs resize the entire frame
and retain horizontal flips/color jitter, removing random crops, RandAugment,
and random erasing. This is an augmentation-bundle comparison, not a controlled
isolation of cropping alone. All retained epoch one; later epochs degraded.

The unweighted full-court run corrected 81 net missed gameplay windows relative
to the first fine-tune (127 to 46), with false positives increasing from 14 to
38. It improved F1 on four validation matches and tied the fifth. Its checkpoint
is a research candidate pending the separate regression evidence; do not infer
production qualification from classifier F1.

Evidence: `/mnt/t/output/skyball_gameplay/human_app_v0_rnd_20260916`.
The directory contains the exact `plan.json`, standalone `comparison.json`,
visual `index.html`, learning curves, code snapshot, and annotation-audit queue.
Full-video gate evidence is under `gate_best`; original-v5 validation and
combined old/new metrics are under `legacy_validation` when complete.

The direct full-court candidate's regression checks subsequently completed:
legacy-v5 F1 improved from 93.988% to 94.296%, and pooled legacy+human F1
improved from 93.734% to 94.405%, all at threshold 0.5. The ten-video contract
passed with threshold 0.15, no gap filling, and 10-second padding on each side:
100% annotated-gameplay, action, and rally retention; zero rejected matched
actions/valid plays; 94/114 target action FPs rejected versus the registered
baseline's 92/114. That temporal policy was selected on the same ten videos.
Completed ledger: `validation/gameplay_human_app_v0_fullcourt_v1.json`.
No production registry or pipeline change was made.

## Authorized two-stage follow-up

The user requested and approved full-court training on the older data followed
by human-only fine-tuning. The controlled follow-up starts from the registered
v5 checkpoint and uses the same v5 dataset that trained it:

1. Adapt on 105,861 legacy training windows from 328 matches for two additional
   conservative epochs. Head LR 2e-5, encoder LR 1e-6, decay after epoch one;
   choose the checkpoint by F1 on the unchanged 15,800-window/27-match v5
   validation set. The short schedule intentionally adapts an already-trained
   checkpoint rather than restarting foundation-model training.
2. Initialize from that checkpoint and repeat the winning four-epoch human-only
   recipe exactly, including seed, augmentations, and five validation matches.
3. Compare against direct human fine-tuning, then evaluate legacy validation
   and the existing ten-match full-video safety/effectiveness contract.

Plan and launch/evaluation controllers:
`/mnt/t/output/skyball_gameplay/human_app_v0_two_stage_20260916`.
Training outputs are `v5_siglip512_fullcourt_adapt_20260916` and
`human_app_v0_siglip512_after_fullcourt_v5_20260916` under
`/mnt/t/output/trn/skyball_gameplay`. Check completion artifacts and logs before
treating this follow-up as complete. No registry promotion is implied.

### Completed two-stage results (September 16)

Both stages and regression evaluations completed. The original v5 run trained
four epochs from pretrained SigLIP-Base weights and retained epoch three. The
new legacy adaptation retained its second epoch; the human stage retained its
first of four epochs. Subsequent human epochs scored 91.228%, 93.590%, and
92.983% F1. Standalone evaluation confirmed the selected checkpoint's score.

| Model | Old v5 validation F1 | Human validation F1 | Pooled F1 |
|---|---:|---:|---:|
| Original model | 93.988% | 91.386% | 93.734% |
| Direct full-court human fine-tune | 94.296% | 95.380% | 94.405% |
| Legacy full-court then human fine-tune | 93.867% | 96.017% | 94.084% |

All window scores use threshold 0.5 and unchanged labels. Old validation has
15,800 windows from 27 matches; human validation has 1,420 windows from five
matches. Exact old-validation sample identities and labels were checked against
the original baseline predictions. Two-stage human precision is 95.652%, recall
96.386%, and ROC-AUC 0.983905, with 33 false negatives and 40 false positives.
Legacy adaptation alone scored 93.111% on human validation before human
fine-tuning; that diagnostic did not select the legacy-stage checkpoint.

The two-stage model passes the unchanged ten-video gate contract with threshold
0.15, no gap filling, and 10-second padding on both sides. It retains all
16,535.239 annotated gameplay seconds, 8,848 confirmed actions, and 1,370 full
rallies, rejecting zero of 8,318 matched actions and 1,395 valid plays. It rejects
94/114 target false positives (20 remaining), matching the direct candidate and
exceeding the original policy's 92/114. Of 63 policies examined, 36 met every
safety condition. This is retrospective selection on the same ten videos.

The staged model improves human F1 by 0.638 points versus direct fine-tuning,
but lowers old-validation F1 by 0.430 points. Thus it does not dominate the
direct model, which retains the better pooled F1. The existing difficult-case
gallery still shows missed serves and huddles; the two previously flagged
timeout annotations remain unchanged. Epoch/experiment selection used human
validation, so these scores are not independent test estimates.

Final ledger: `validation/gameplay_human_app_v0_two_stage_v1.json`.
Final comparison, review gallery, checkpoint hash, and full gate evidence:
`/mnt/t/output/skyball_gameplay/human_app_v0_two_stage_20260916/{summary.md,index.html,result.json}`.
No registry promotion, deployment, or higher-resolution training was performed.

### Selected production candidate and portable qualification

The user selected the two-stage checkpoint with threshold 0.15, zero merge gap,
and 10 seconds of padding per side. The canonical primary-video audit retains
all 8,848 confirmed events and admits 46.42494% of total match time outside the
three-second event margins. Original production at 0.20/30s retains every event
with 52.48766% extra footage. At the same 0.15/10s policy, the original model
excludes 13 events with 44.76645% extra footage. See
`workflows/gameplay_gate_validation.md` for definitions and evidence.

ONNX parity passed. TensorRT 11.1 strict FP32 (TF32 disabled) passed on 41 real
windows at batch size 8, with zero threshold disagreements and maximum gameplay
probability difference 3.516674e-6. Disable TF32 in both the engine and PyTorch
reference when reproducing this strict numerical check.

Portable model, source/split hashes, exact manifests, training commands, epoch
metrics, both validation sources, and `TRAINING_PROGRESSION.md` are assembled at
`/mnt/t/output/skyball_gameplay/model_publication_human_v0_staged_20260916/human_v0_siglip512_fullcourt_staged_20260916`.
The corresponding immutable Drive destination is
`googdrive:skyball/checkpoints/gameplay/human_v0_siglip512_fullcourt_staged_20260916`.
The registry's `recommended_candidate` records this selection separately from
the deployed model. Full end-to-end acceptance against the current pipeline
release remains required before activation; portable qualification is complete.
