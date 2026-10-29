# Event-refined gameplay: deferred experiment plan

Date: 2026-09-20. Status: planning only; wait for GPU resources and a user
instruction to resume. Do not launch inference, training, or a GPU waiter from
this plan. The September 19 rectangular experiment's evaluation remains stopped.

## Objective and hypothesis

Keep the current human/VLM phase annotations as coarse supervision, then use
detected volleyball actions to refine gameplay intervals. The requested rule is:

`fine_gameplay(t) = coarse_gameplay(t) AND within_2_seconds_of_an_accepted_event(t)`

Between-play pauses inside coarse gameplay would become negative. Actions in
warmup remain negative because they are outside coarse gameplay. The hypothesis
is that visually inactive pauses currently labeled positive contribute to
confusion. This is not yet established as the cause of the model's errors.

These are two stages of label construction, separate from the existing two-stage
model-training recipe. Preserve original annotations and write derived labels
alongside them; detector predictions are pseudo-labels, not confirmed truth.

## Frozen starting point

- Use the already imported September 19 v2 snapshot: 175 source videos, split
  into 155 training and 20 validation videos. Inventory this snapshot explicitly;
  do not silently include later Drive changes.
- Dataset: `/mnt/t/data/vball/skyball/gameplay/human_app_v2_val20_20260919`.
  Current counts: 35,193 training windows and 4,828 validation windows.
- Split/provenance:
  `/mnt/t/output/skyball_gameplay/human_app_v2_expanded_val20_b128_20260919_r1/split_plan.json`.
- The original five validation matches have frozen human labels; the added
  fifteen have model-reviewed labels pending human acceptance. Report them
  separately. Retain the frozen original-five annotation versions.
- Raw full-match videos live under
  `/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user`.
  Workbook timestamps refer to these videos, never `images_full` play clips.
- Primary classifier comparator: the square full-frame v2 batch128 checkpoint
  `human_app_v2_val20_siglip512_stage2_b128_20260919_r1/best_siglip512_app_v2_val20_b128.pth.tar`
  under `/mnt/t/output/trn/skyball_gameplay`. Also retain the September 16 model
  as a comparator. Record hashes before reuse.

## 1. Inventory and freeze detector inference

When resources are available, check GPU ownership and use explicitly available
devices. Start with a small pilot to measure runtime and inspect labels, then
process all 175 videos with resumable per-video jobs in tmux.

Before launching:

1. Resolve the current approved detector checkpoint, inference implementation,
   preprocessing, action class map, confidence thresholds and temporal
   deduplication settings. Record exact versions/hashes; do not guess a model
   path or invent a CLI before checking the detector repository.
2. Audit existing event outputs for possible reuse. Reuse only when video
   identity, full-video time coordinates, checkpoint and inference settings
   match. A gameplay-filtered or partial run is not complete evidence.
3. Audit detector training overlap with the held-out videos where provenance is
   available. Such overlap limits independence of detector-derived validation.
4. Run detection without the current gameplay classifier gating its input or
   outputs. Otherwise existing classifier false negatives become invisible.
   Avoid downstream filters that themselves depend on gameplay predictions.

Run over full videos, retaining raw action detections outside gameplay for the
warmup audit. Use volleyball action events, not player/ball presence or referee
gestures. Freeze the eligible action classes and confidence policy after the
training-source pilot. Preserve scores and discarded detections for auditing.

Record exact original-video frame indices and timestamps, source FPS/time base,
any chunk offsets or transcoding maps, source identity, runtime and completion
coverage. An empty successful event list must be distinguishable from missing or
failed inference. Never label unprocessed footage negative.

## 2. Construct the proposed labels

For each coarse gameplay interval G, select accepted events whose timestamps
fall inside G. Union their `[event_time - 2s, event_time + 2s]` neighborhoods,
clipping each union to G and the recording bounds. An event outside G must not
spill a positive neighborhood into G. Fix endpoint conventions in source-frame
coordinates and test them.

Keep the original phase plus a separate fine label/reason:

| Region | Proposed fine label | Provenance |
|---|---|---|
| Inside coarse gameplay, within ±2s of an accepted event | Positive | Detector-derived |
| Inside coarse gameplay, farther from every accepted event | Negative candidate | Absence of detection; requires audit |
| Prematch/warmup, timeout, set break, postmatch | Negative | Original phase label |
| Missing inference, bad timing, or unresolved annotation conflict | Ignore | Explicit reason |

No events does not prove no play. Review coarse gameplay intervals with no
accepted events, long gaps between events, low-confidence near-misses, and
visually active regions assigned negative. Preserve the strict-rule labels for
analysis, but keep unresolved cases out of training until reviewed. Report the
amount excluded; do not hide it inside a cleaned dataset.

### Four-frame target policy

Inputs remain frames at `t, t+1, t+2, t+3`. Proposed primary scalar target:
evaluate the fine label at the center timestamp `t+1.5`. Store all four frame
labels and the fraction of the window covered by positive intervals as well.
Mixed windows are visible in review rather than automatically called ambiguous
or silently dropped. Validate that center supervision is learnable from the
sampled frames before locking this policy.

A positive center does not mean all four seconds are positive. Document how
scores become admission ranges during full-video evaluation. If runtime admits
the whole window, measure that expansion explicitly; do not claim its effective
support is exactly ±2s. Do not apply a five-second exclusion margin at every
new event boundary: it could remove these short positive intervals entirely.

## 3. Review a pilot before scaling label extraction

Select about five training sources using recording/annotation metadata: clear
endline view, wide or angled court, substantial warmup, long between-play pauses,
and difficult visibility. Do not choose all pilot examples from validation or
tune event thresholds on validation performance.

Produce one clickable page per pilot source with aligned tracks for coarse
phase, detected events/confidence, proposed fine label and available classifier
scores. Include four-frame thumbnails opening full images, plus timestamped
8–12-second video context for motion inspection.

Review buckets:

- Newly negative between-play pauses that look correctly relabeled.
- Newly negative real action: detector misses or gaps inside an ongoing rally.
- Serve preparation and rally starts/ends near the ±2s boundaries.
- Warmup detections excluded by the coarse mask, including multiple-ball practice.
- Suspected original annotation errors and windows with mixed frame labels.

Include both a representative random sample and targeted difficult examples;
record their selection separately. Summarize per match and pooled: duration
reclassified, positive/negative window counts, zero-event intervals, and reviewed
error counts with denominators. Human review distinguishes active play,
between-play inactivity and uncertain evidence. Freeze the policy after review
before making the complete training dataset.

## 4. Build the derived dataset and run a controlled training comparison

Write versioned outputs under `/mnt/t/output/skyball_gameplay/<study>` and any
new image dataset/manifests under
`/mnt/t/data/vball/skyball/gameplay/<new_version>`. Do not edit existing v2
annotations, images, split manifests or completed runs.

Include source/window IDs, original and fine labels, label provenance, raw-video
timestamps/frame coordinates, nearest-event distance, supporting event IDs,
detector/config hashes, confidence, review state and ignore reason. Preserve
source splits and original class names. Existing JPEGs can be reused read-only.

First ablation: retain the same sampled windows, square full-frame512 geometry,
existing flips/color jitter, stage-one initializer, four-epoch schedule, and
effective batch128. Change supervision only. Audit the new class balance first;
the existing 20-second gameplay stride may sparsely cover short event intervals.
If event-centered or denser sampling is needed, make it an explicit second
ablation with fixed validation, rather than attributing its effects to labels.

Publish native live training loss/accuracy and epoch validation F1/loss to
TensorBoard. Save validation-window scores during each validation pass so a
second GPU pass is unnecessary for cohort reports and threshold sweeps. Use
human-reviewed fine validation for model selection where available; report
teacher-label agreement separately. Save checkpoints and record the selected
epoch and selection rule.

## 5. Evaluate the intended behavior and safety

The fine target differs from the original phase target. A drop in original-label
F1 may represent deliberate rejection of pauses, so report both targets without
calling them directly comparable measures of success.

Compare old and candidate predictions on identical held-out videos, reporting:

- Fine-label precision/recall/F1 on independently reviewed windows, and separate
  agreement with detector pseudo-labels; include original-five/added-fifteen views.
- Original phase-label F1 on frozen human and legacy validation, with the target
  mismatch explained and event-adjacent versus between-play errors separated.
- Confirmed action retention, missed actual rallies, and warmup false admissions.
- On continuously scored recordings: durations of missed-action gaps and false
  admission bursts. Sparse windows alone cannot measure temporal filtering.

For any candidate being considered further, use the canonical full10 primary
videos and remapped GT under
`/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1`.
Report confirmed events excluded and duration-weighted extra footage outside
±3s of confirmed actions. This evaluation margin is distinct from the ±2s
training-label rule. Report per match, pooled, threshold, padding per side,
merge gap and runtime; compare models at identical settings first.

Use confirmed GT as the independent action reference wherever available. A
classifier reproducing its teacher's missed detections is not evidence of safety.
Check full-rally retention and target-FP rejection, matched-action retention and
valid-play retention per `gameplay_gate_validation.md`.

**Contract issue to resolve explicitly:** the current gate requires retention of
all annotated gameplay and full rallies. A fine action mask deliberately dropping
between-play footage may conflict with that requirement, and ±2s event unions
may also leave gaps inside rallies. Keep the existing contract intact. Evaluate
whether a separately reported temporal admission policy can satisfy it; otherwise
retain this as an experimental action-activity signal. Any proposal to change its
production purpose or acceptance contract needs explicit review and a separately
versioned contract. No automatic replacement of the coarse gate is implied.

## Resume checklist and expected artifacts

- [ ] Confirm GPUs and freeze detector/video/annotation identities.
- [ ] Run pilot detector inference without gameplay gating; estimate full-run cost.
- [ ] Review timelines/clips and freeze event and window-target policies.
- [ ] Process all 175 frozen sources with validated per-video completion records.
- [ ] Write derived manifests and label-change/coverage audits; review dataset.
- [ ] Run the label-only training comparison with live metrics and saved val scores.
- [ ] Complete both classification and full-video safety comparisons before recommending.

Deliverables: run manifest, raw events and inference coverage, derived intervals
and window labels, review site and annotations, audit summary, experiment config,
checkpoints, TensorBoard logs, and a comparison report stating open limitations.
