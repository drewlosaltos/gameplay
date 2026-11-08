# Gameplay Gate Validation

## Required second validation source

After every completed gameplay training/fine-tuning candidate, evaluate both
the held-out classification dataset and the canonical ten full matches. The
ten-match source must report the user's two operational metrics: confirmed
events excluded, and extra admitted footage as a percentage of the full match.
Use the exact definitions and input bindings in the production-frame audit
section below. Report per match and duration-weighted aggregate values with
threshold, padding per side, merge gap, and runtime. Compare identical model
settings before attributing differences to model weights; selected policies
may also be compared separately. Persist both validation sources in the
candidate result ledger and any recommended model registry entry.

Zero excluded events is the first priority. A classification F1 increase alone
does not complete validation. These ten matches are used as validation for
model/policy assessment, not claimed as an independent test population.

Use this workflow to validate a speculative gameplay gate against the ten
SkyBall evaluation matches and the current pipeline action outputs. This does
not integrate the classifier into the pipeline.

## Contract

The canonical contract is
`validation/gameplay_gate_eval10_v2.json`. It fixes both the evaluation
population and six promotion checks:

1. 100% of margin-expanded annotated gameplay seconds are retained.
2. 100% of confirmed synchronized GT actions are retained.
3. 100% of synchronized rallies are fully covered.
4. At least 92 of the 114 target action false positives are rejected (at least
    80.7018%). A target FP is an unmatched predicted action more than three
    seconds from every confirmed synchronized GT action.
5. Zero currently matched true actions are rejected.
6. Zero currently valid predicted plays are rejected.

The fixed population contains ten matches, 8,848 confirmed actions, 1,370
rallies, 114 target action FPs, 8,318 matched true pipeline actions, and 1,395
valid predicted plays. A membership or denominator change returns
`invalid_population` rather than passing or failing a model.

The current production policy (Qwen V8, `v8_val48_qwen672x384_single_stage_fp8_20261006_r3`)
uses gameplay score threshold `0.9`, merge gap `0` and 5 seconds of padding on each side, sampled
at 1 Hz. On this contract it rejects 94 of 114 target FPs, excludes 0 of 8,848 events, and admits
39.6748% outside-event footage. It was selected on these same ten matches, so this gate is a
non-regression contract, not an unbiased estimate of unseen-match transfer. Earlier baselines
(SigLIP-512 at 0.2 / 30 s; Qwen v5 at 0.5 / 20 s) are in `HISTORY.md` and
`docs/history/model_registry_history.yaml`.

## Primary R&D Selection Rule

Evaluate every new gameplay checkpoint with two complementary views:

1. Report clean val48 classifier F1 at threshold 0.5
    (`validation/gameplay_indoor_v8_val48_clean_v2.json`) for the selected epoch.
2. Score all ten full videos and search temporal policies. Discard every policy
    that misses any annotated gameplay second, confirmed GT action, full rally,
    matched true pipeline action, or valid predicted play. Among the remaining
    safe policies, select the one rejecting the most target action FPs. Use
    invalid-play rejection and excluded video duration only as secondary
    diagnostics or tiebreakers.

The model registry must contain both result families. A higher classifier F1
does not supersede a model that has better safe target-FP rejection unless the
new model also passes the full-video non-regression contract.

## Generate Evidence and Apply the Gate

The exact commands for a Qwen checkpoint are in `REPRODUCE.md` ("Full10 gate and
promotion"), and they run in the `skyball-gameplay-train` env (`skyball-prod-infer` on the training host) with the pipeline repo on
`PYTHONPATH`:

1. `scripts/score_gameplay_pipeline_inputs.py` scores the ten canonical inputs in FP32 with
   production RGB preprocessing (Qwen letterbox supported since `879eebe`), one shard per GPU.
2. `scripts/make_gate_score_view.py` lays the scores out for the gate scripts.
3. `scripts/evaluate_gameplay_pipeline_fp_impact.py --threshold .9 --merge-gap-sec 0 --padding-sec 5`
   writes the retrospective impact report.
4. `scripts/validate_gameplay_gate.py --contract validation/gameplay_gate_eval10_v2.json`
   applies the contract.
5. `scripts/evaluate_gameplay_event_coverage.py --candidate-threshold .9 --candidate-padding-sec 5`
   reports events excluded and outside-event footage.

To search a new policy, use the safety-first rule above and record the full grid. The
production search is `workflows/qwen_v8_production_20261007/search_single_stage_policy_when_ready.py`:
17 thresholds x 8 paddings x 2 gaps.

`validate_gameplay_gate.py` exit codes are:

- `0`: all six metrics pass and the population matches;
- `1`: one or more model metrics fail;
- `2`: evaluation membership or a fixed denominator changed.

Never relax or silently rewrite the existing contract for a changed evaluation
set. Add a new versioned contract and record its new population and baseline.

## Runtime Qualification

The production gameplay runtime is PyTorch, not ONNX or TensorRT. The registry records
`onnx_status: not_exported_pytorch_runtime_selected`. A candidate is published as an FP32
`model.safetensors` bundle that the pipeline's `skyball_pipeline.gameplay_qwen` runtime loads.
The input is normalized RGB `[batch, 4, 3, 384, 672]`, letterboxed with gray128 padding. The
outputs are two-class `logits` and the scalar `gameplay_score`. Build the bundle with
`scripts/export_qwen_gameplay_bundle.py`. Its `--parity-scores` option re-scores the first
120 s of one full10 match through the pipeline runtime and requires a max score difference
below 0.005 with no decision changes at the threshold. FP8 applies to training only.

The classifier threshold used for window-level F1 (0.5) and the temporal gate threshold
selected for safe full-video filtering (0.9) are separate operating points and must not be
interchanged.

The SigLIP-era ONNX/TensorRT qualification (`export_gameplay_onnx.py`, `build_gameplay_trt.py`)
and the September 16, 2026 production-frame event admission audit are in `HISTORY.md`.
