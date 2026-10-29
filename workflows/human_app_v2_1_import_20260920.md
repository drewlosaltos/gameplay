# Gameplay app v2.1 preparation (September 20, 2026)

Internal version **v2.1** is a frozen refresh of the Drive **v2** folder, not a
new remote dataset version. Source updated at2026-09-20T20:45:12Z, marked complete:
`googdrive:skyball/datasets/gameplay/indoor/skyball-gameplay-finetune/v2`.

The prior import captured175 sources while v2 was in progress. The revised export
contains248 eligible sources:75 additions and2 previously imported training
sources removed by its confirmed exclusion policy. It includes198 model-reviewed
sources and50 inherited human-labeled sources. Model-reviewed labels still await
human acceptance; do not describe the whole refresh as human-labeled.

The20-match validation split stays frozen, including all4,828 window records and
19,312 JPEGs. Training membership becomes228 videos. Retain the JJC validation
source: its exclusion is proposed, not confirmed. Source aliases and supplemental
recordings are archived but not admitted as independent matches, particularly
when related to held-out matches.

One retained training match, `vid_mtqnm9s81066e9`, changes0–35s from gameplay to
prematch. Resample only this corrected source and75 new sources; hardlink all
unchanged retained samples. Excluded old sources are omitted only from the new
export; original datasets and videos remain intact.

Paths:

- Work/evidence: `/mnt/t/output/skyball_gameplay/human_app_v2_1_import_20260920_r1`
- Dataset: `/mnt/t/data/vball/skyball/gameplay/human_app_v2_1_val20_20260920`
- Frozen base: `/mnt/t/data/vball/skyball/gameplay/human_app_v2_val20_20260919`
- Review site: `http://atao-9355:6011/` (pages appear as sources finish).

Reuse the original canonical sampler/extractor: four frames at1Hz, gameplay
stride20s and negative stride10s, centered transition exclusion zones20s for
prematch/gameplay and5s otherwise; original ffmpeg fps2/select timing;1024-long-
edge PIL Lanczos JPEG88. Use CPU decoding and libx264 review encoding only.
Drop complete newly sampled windows extending past visual EOF, separately
recording audio-tail exclusions. Never alter frozen validation to repair a source.

Preparation is resumable in the named tmux sessions
`gameplay_v2_1_download_20260920` and `gameplay_v2_1_prepare_20260920`. The controller
contains no training or detector launch. It processes only checksum-verified,
completed downloads; final manifests and readiness require full source/frame
coverage, boundary checks and exact frozen-validation parity. Inspect
`status.json`, `dataset_validation.json` and `dataset_complete.sentinel` before
calling this dataset ready. No automatic GPU waiter or training is authorized.

After import, `training_preparation.json` records an unlaunched square512 SigLIP
stage-two/batch128 configuration using the previous stage-one initializer. Assign
a new training run name and check resources when the user resumes. Detector-event
label refinement remains the separate deferred plan in
`event_refined_gameplay_plan_20260920.md`.
