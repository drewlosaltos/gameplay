# Reproducing the production gameplay model

Prod model: **`v8_val48_qwen672x384_single_stage_fp8_20261006_r3`**. It's a four-frame Qwen3.6
vision encoder at 672x384 with temporal mean pooling and a binary head, trained in one stage
from pretrained Qwen. We kept epoch 3 of 4, with clean val48 F1 0.961224. The pipeline runs
it at threshold 0.9, with 5 s of padding per side and merge gap 0.

```bash
python scripts/reproduce.py recipes/v8_val48_qwen672x384_single_stage_fp8_20261006_r3.json \
  --output-dir /mnt/t/output/trn/skyball_gameplay/<new_versioned_run>
```

The recipe (`recipes/<model_id>.json`, schema `skyball_repro_recipe/v1`) lists every input and
its hashes, plus the environment, hardware, selection rule, expected metrics and tolerance.
The hyperparameters live in `config/indoor_canonical.json`, whose `args` are exactly the flags
prod ran with. `tests/test_reproduce.py` checks that the recipe and that config agree.

## What you get

| Item | Where | Identity |
|---|---|---|
| Published FP32 pipeline bundle | `googdrive:skyball/checkpoints/gameplay/v8_val48_qwen672x384_single_stage_fp8_20261006_r3/` | `artifact.json` `0a2459d7…`, `model.safetensors` `fe1bf189…` |
| Training checkpoint (epoch 3) | same folder, `training_checkpoint/best_qwen672x384_v8_single_stage_4gpu.pth.tar` | `67f342a3…` |
| Training data | `googdrive:skyball/datasets/derived/gameplay/v8_val48_trainclean_overlay_20261006_r4` (19 zips, 154 GiB) | `windows.jsonl` `8b154d8d…` |
| Base weights | Hugging Face `Qwen/Qwen3.6-35B-A3B-FP8` @ `95a723d08a9490559dae23d0cff1d9466213d989` (only the vision tower is used) | per-file sha256 in the recipe |
| Original run | `/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_single_stage_4gpu_20261006_r3` (training host only) | |

## Prerequisites

- **Hardware:** 4x RTX 5090 (32 GB). Prod took 5.6 h wall time, at about 0.30 s per
  iteration and 14,928 iterations per epoch per GPU. The trainer needs exactly four GPUs, and
  the effective batch is 16.
- **Disk:**
  - About 154 GiB of staged zips plus 154 GiB unzipped.
  - 2.9 GB of Qwen weights.
  - About 11 GB per run (two 5.4 GB checkpoints).
- **Drive:** an rclone remote named `googdrive` that can read `skyball/`.
- **Hugging Face:** the Qwen repo is public, so `hf download` needs no token (set `HF_TOKEN`
  for higher rate limits). Weights land in `$HF_HOME/hub` (default `~/.cache/huggingface/hub`).
  `SKYBALL_QWEN_SNAPSHOT=<dir>` points at a snapshot elsewhere. The config and old checkpoints
  record `/home/atao/.cache/huggingface/...`. Both the launcher and checkpoint loading re-root
  that path under your hub cache, or use `SKYBALL_QWEN_SNAPSHOT` (`skyball_gameplay/data_root.py`).
- **Environment:**

  ```bash
  conda env create -f environment.yml          # Python 3.12.13 + requirements-lock.txt (torch 2.11.0, CUDA 13.0)
  export SKYBALL_TORCHAO_OVERLAY=/path/to/torchao_overlay
  conda run -n skyball-prod-infer pip install --no-deps --target "$SKYBALL_TORCHAO_OVERLAY" \
    torchao==0.17.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  ```

  - **TorchAO:** prod ran torchao **0.17.0+cu128** from a `PYTHONPATH` overlay on top of torch
    **2.11.0+cu130**. On the training host the overlay is
    `/mnt/t/output/skyball_gameplay/v8_qwen_fp8_prototype_20261006_r1/site`, which symlinks
    the `pt5090new` env's torchao. That is the default when `SKYBALL_TORCHAO_OVERLAY` is
    unset, and the wheel above has identical file hashes.
  - **The CUDA mismatch:** the recipe only uses torchao's tensorwise `Float8Linear`, which runs
    on `torch._scaled_mm`. Its cu128 C++ kernels are unused and untested on cu130. Keep this
    pairing so you match prod; don't "fix" it.
  - **What `requirements-lock.txt` leaves out:** its header lists them. `skyball_pipeline`
    (full10 and export only) and the system ffmpeg are installed separately.
- **Data root:** `SKYBALL_DATA_ROOT` (default `/mnt/t/data/vball/skyball/unified_dataset`).
  `SKYBALL_STAGING_ROOT` (default `/mnt/t/data/vball/skyball/unified_staging`) holds the zips.
  `SKYBALL_REPRO_ROOT` (default `/mnt/t/output/skyball_repro`) receives Drive files that only
  `--export` needs.

Run everything from the repo root inside the env, in tmux, after checking `nvidia-smi`.

## Stages

With no stage flag, `reproduce.py` runs fetch, check, train and evaluate. Each stage can be
run on its own:

| Flag | What it does | GPU |
|---|---|---|
| `--fetch [--print-only]` | Downloads the missing inputs: `hf download` for the 3 Qwen files, then `rclone copy` + `sha256sum -c SHA256SUMS` + `unzip -n` for the dataset. Anything already present is verified and skipped. `--print-only` prints the commands without running them. Add `--export` to also fetch the prod bundle. | no |
| `--check` | Checks package versions against `env.require`, that the TorchAO overlay is present, input sha256s (Qwen files, `windows.jsonl`, `summary.json`), and that the recipe and config agree. Also runs the val48 split audit (membership, overlap, known re-encodes) and prints the dry-run training command. | no |
| `--train --output-dir <new>` | Refuses an existing directory. Runs `scripts/train_qwen_canonical.py` with the canonical config. Writes `canonical_plan.json`, `reproduction_provenance.json` (git SHA and dirty flag, versions, input identities, resolved recipe), `code_snapshot/` and `launcher.log`. `--gpus a,b,c,d` overrides the GPUs. | 4 |
| `--evaluate --output-dir <run>` | Applies the select rule to `epoch_metrics.jsonl`, re-scores the selected checkpoint on val48 (`python -m skyball_gameplay.evaluation`, FP32, first GPU) and writes `reproduction_report.json`. `--skip-rescore` skips the GPU re-score, and `--report <path>` writes the report elsewhere. | 1 |
| `--export --output-dir <run> --gate-root <dir> --bundle-dir <new>` | Builds the pipeline bundle (see "Export") after the full10 gate passes. | no |

Exit codes: 0 means OK, 1 means the check failed, and 2 means the report is FAIL.

## Reading the report

`reproduction_report.json` has one row per check, each `PASS`, `FAIL` or `SKIPPED`. The
overall `status` is PASS when nothing fails.

| Epoch | Expected clean val48 F1 | Precision | Recall |
|---:|---:|---:|---:|
| 1 | 0.957698 | 0.960743 | 0.954672 |
| 2 | 0.957536 | 0.953774 | 0.961328 |
| **3 (selected)** | **0.961224** | 0.972411 | 0.950291 |
| 4 | 0.960236 | 0.972686 | 0.948100 |

- **Tolerance:**
  - Each epoch's F1 must be within ±0.005 of the table.
  - Every epoch must score exactly 16,281 windows.
  - The selected epoch must be 3 or 4 (they're within 0.001).
  - The selected epoch's F1 must be within 0.005 of 0.961224.
  - The standalone re-score must be within 0.005 of the selected epoch's in-training F1.
- **Why 0.005:** prod ran once, so seed-to-seed noise is unmeasured, and FP8 + compile + NCCL
  are not bitwise deterministic. 0.005 is about 2.5 binomial standard errors. It is also
  larger than the epoch 3/4 gap (0.0010) and the batch 8 vs 16 pilot gap (0.0023).
- **Not checked:** the full10 gate is not part of PASS/FAIL. Run it next.

Example: `--evaluate --skip-rescore` pointed at the original prod run passes every check.

## Full10 gate and promotion

Every candidate also needs the canonical full10 gate. That's the second validation source,
and promotion is decided on it. Its inputs aren't on Drive. They are bound by absolute path
in `validation/gameplay_event_coverage_full10_v1.json` and `skyball_gameplay/full10.py`, so
run it on the training host (or recreate the same paths):

- `PRIMARY=/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1`: the
  ten videos with synced GT. Two of them are 30 fps transcodes in
  `video_transcodes_30fps_keyframes_20260806_v1`.
- `SUITE=/mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1`: the
  pipeline manifests and timing that define 1 Hz sampling.
- `FP_SUITE=/mnt/t/output/skyball_pipeline/evaluation_suite_serve_primary_e2e_cached_detector_v1_20260717`:
  the pipeline actions used for target false positives.

```bash
export PYTHONPATH=$PWD:/path/to/pipeline        # Skyball-AI/pipeline (1b509cb here)
S=/mnt/t/output/skyball_gameplay/<run>_full10; CK=<run>/best_qwen672x384_v8_single_stage_4gpu.pth.tar
for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/score_gameplay_pipeline_inputs.py \
  --shard $i --num-shards 4 --checkpoint $CK --pipeline-root $SUITE --output-root $S/full10_scores & done; wait
python scripts/make_gate_score_view.py --scores $S/full10_scores --output $S/gate_score_view
python scripts/evaluate_gameplay_pipeline_fp_impact.py --evaluation-root $PRIMARY --study-root $S/gate_score_view \
  --pipeline-root $FP_SUITE --output-dir $S/impact --threshold .9 --merge-gap-sec 0 --padding-sec 5
python scripts/validate_gameplay_gate.py --report $S/impact/pipeline_fp_impact.json \
  --contract validation/gameplay_gate_eval10_v2.json --output $S/impact/validation_gate_result.json
python scripts/evaluate_gameplay_event_coverage.py --pipeline-root $SUITE --candidate-scores $S/full10_scores \
  --output-root $S/coverage --candidate-threshold .9 --candidate-padding-sec 5
```

The coverage script on main always uses merge gap 0.

**Expected (prod):**

- Gate `pass`.
- 0 of 8,848 confirmed events excluded, and all 1,370 rallies fully kept.
- 94 of 114 target false positives rejected.
- Outside-event footage 39.6748% of total match time.

These come from `impact/pipeline_fp_impact.json` `aggregate`, `coverage/metrics.json`
`aggregate.candidate_selected_policy` and `validation_gate_result.json` `status`. Prod's own
files are in the Drive bundle's `evidence/`. Running the last four commands on prod's cached
scores reproduces those aggregates exactly.

**Scoring cost:** about 1,930 GPU-seconds summed over the matches.

**Promotion:** compare against prod on full10, with zero excluded events first and then
target-FP rejection and outside-event footage. A regression usually means no promotion. A
new policy must be searched safety-first, with the grid recorded, as in
`workflows/qwen_v8_production_20261007/search_single_stage_policy_when_ready.py`. See
`workflows/gameplay_gate_validation.md` for the contract and its definitions. The ten
matches selected the policy, so treat this as retrospective validation, not a test set.

## Export

```bash
python scripts/reproduce.py recipes/v8_val48_qwen672x384_single_stage_fp8_20261006_r3.json --fetch --export --print-only   # prod bundle = template
python scripts/reproduce.py recipes/v8_val48_qwen672x384_single_stage_fp8_20261006_r3.json --export \
  --output-dir <run> --gate-root $S --bundle-dir /mnt/t/output/skyball_gameplay/<release>/<model_id>
```

Export runs `scripts/export_qwen_gameplay_bundle.py`, which:

- writes the FP32 `model.safetensors` from the checkpoint;
- copies `qwen_vision_config.json` from the prod bundle;
- writes `checkpoint_metadata.json`, `training_config.json`, `training_source.tar.gz`
  (`code_snapshot/`), `gameplay_operating_points.json` (0.9 / 5 s / 0) and the full10 evidence;
- refuses a failing gate, and hashes every file into `artifact.json`.

On the prod checkpoint it reproduces `model.safetensors` byte for byte (`fe1bf189…`). For
runtime parity, run the script directly with
`--parity-scores $S/full10_scores/eval_club_005_pac_balboa`. That needs `skyball_pipeline`
and a GPU, and requires 117 windows to differ by less than 0.005, with no decision changes at
0.9. Pipeline catalog/release updates are a pipeline-repo step. The original ones are in
`/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1/`.

## Code provenance

- **What prod ran:** prod trained from `<run>/code_snapshot/skyball_gameplay` (also
  `training_source.tar.gz` on Drive). It's a copy of the main working tree from 2026-10-07
  00:27, and no git SHA was recorded.
- **Training path:** `training`, `data`, `qwen_vision`, `acceleration`, `preprocessing` and
  `full10` are byte-identical to `07d6eb7` (the promotion commit, the recipe's
  `code.min_commit`).
- **Eval-only difference:** the snapshot's `evaluation.py`/`modeling.py` also carry the
  `--image-size` override, committed later in `879eebe`.
- **Later main:** it adds the rally-net features (`--keep-epoch-checkpoints`,
  `covered_event_ids`). They're opt-in and inactive for this recipe, and
  `skyball_gameplay.data_root` adds the path resolution.
- **Launcher:** prod was launched by
  `workflows/qwen_v8_production_20261007/launch_single_stage_v8_when_ready.py`, not by
  `train_qwen_canonical.py` (which was written from it the same day). The two produce
  identical trainer arguments, apart from the dataset path (moved to `unified_dataset` on
  2026-11-05, same `windows.jsonl`) and the output/port.
- **One-off scripts:** the run's val48, full10, policy-search and export scripts are kept
  verbatim in `workflows/qwen_v8_production_20261007/`. The evidence and decisions are in
  `workflows/qwen_v8_production_20261007.md`, and older recipes are in `HISTORY.md`.
