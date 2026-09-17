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

The current baseline temporal policy uses a gameplay score threshold of `0.2`,
does not fill inactive gaps, and pads each retained interval by 30 seconds on
both sides. It was selected on these same ten matches, so this gate is a
non-regression contract, not an unbiased estimate of unseen-match transfer.

## Primary R&D Selection Rule

Evaluate every new gameplay checkpoint with two complementary views:

1. Report combined-validation classifier F1 and its threshold on the fixed
    window manifests.
2. Score all ten full videos and search temporal policies. Discard every policy
    that misses any annotated gameplay second, confirmed GT action, full rally,
    matched true pipeline action, or valid predicted play. Among the remaining
    safe policies, select the one rejecting the most target action FPs. Use
    invalid-play rejection and excluded video duration only as secondary
    diagnostics or tiebreakers.

The model registry must contain both result families. A higher classifier F1
does not supersede a model that has better safe target-FP rejection unless the
new model also passes the full-video non-regression contract.

## Generate Evidence

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

## Apply the Gate

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

## Portable Runtime Qualification

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

## Production-frame event admission audit (September 16, 2026)

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
