# Gameplay Gate Validation

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
