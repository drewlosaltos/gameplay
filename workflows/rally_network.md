# Rally network (baseline recipe, 2026-11-05)

A binary rally / no-rally classifier with the prod gameplay architecture. Gameplay
admits whole stretches of play; the rally net also rejects the dead time between
rallies inside them. Priority: never miss a real GT touch; disagreement right at
the rally/no-rally border is acceptable.

Recipe: `config/rally_canonical.json`. Full study write-up:
`/mnt/t/output/skyball_gameplay/rally_study_20261105/RESULT.md`.

## Labels and data

- Source: human sparse-v4 events under the data root,
  `derived/detector/pseudo/sparse_v4_xl_dense_fullcls_ctx12_active6_stable_ballactor_h085_low035_deduptrackids_human_events/<cut>/events.json`.
  Only cuts whose `events.json` has a `source` field (human events) are used: 77
  cuts, each a distinct match. All event skills count, including floor_bounce,
  ball_in_net and net_touch.
- A window of frames at t, t+1, t+2, t+3 s (1 Hz, from `images_full`) is `rally`
  when its centre t+1.5 s is within 2 s of any event. Natural class ratio.
- Each window lists the events inside its span (`covered_event_ids`), which
  training validation uses to report GT events missed.
- Split by cut: 12 validation cuts (seed 20261105; 89 min, 769 events), 65 train.

```bash
conda run --no-capture-output -n skyball-prod-infer python scripts/build_rally_sparse_dataset.py \
  --output derived/gameplay/rally_sparse_v4_m2_<date>_r1
```

A relative `--output` goes under `SKYBALL_DATA_ROOT`. The reference dataset
`/mnt/t/data/vball/skyball/gameplay/rally_sparse_v4_m2_20261105_r2` (train 28,481
windows, 30% rally; val 5,308, 26% rally) predates that convention; rebuilding
reproduces its `windows.jsonl` byte for byte.

## Training

The canonical gameplay recipe with the rally dataset, model weights initialized
from prod gameplay v8, per-GPU batch 4 with gradient accumulation to effective
batch 16, 4 epochs, and a checkpoint kept for every epoch:

```bash
PYTHONPATH=/mnt/t/output/skyball_gameplay/v8_qwen_fp8_prototype_20261006_r1/site \
  conda run --no-capture-output -n skyball-prod-infer python scripts/train_rally_from_prod.py \
  --gpus 0,1 --output-dir /mnt/t/output/trn/skyball_gameplay/rally_<tag>_<date>_r1
```

About 25 min per epoch on two GPUs. Use `--dry-run` to print the command.

## Selection and evaluation

Validation logs `gt_events_missed` (at 0.5, 0.3, 0.1) per epoch next to F1, but
769 events, most misses in one match, is too small to select on. Neither
validation F1 nor held-out misses picked the best epoch in the study. Select on
full10:

```bash
PYTHONPATH=$PWD:/home/atao/vsdevel/skyball/pipeline conda run --no-capture-output -n skyball-prod-infer \
  python scripts/score_gameplay_pipeline_inputs.py --shard 0 --num-shards 2 \
  --checkpoint <run>/epoch2_rally_qwen672x384_from_prod_v8.pth.tar \
  --pipeline-root /mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1 \
  --output-root <scores_dir>   # and shard 1 on a second GPU
PYTHONPATH=$PWD:/home/atao/vsdevel/skyball/pipeline conda run -n skyball-prod-infer python scripts/evaluate_rally.py \
  --source full10 --scores epoch2=<scores_dir> --policy 0.5:0 --output <report.json>
```

- GT events missed: a confirmed synchronized event whose frame is not admitted.
  A window scoring at or above the threshold admits [start - pre, end + post]
  (prod gameplay convention).
- Dead time rejected: share of frames more than 2 s from every event that are
  not admitted.
- Select the fewest misses at threshold 0.5 with no padding, then the most dead
  time rejected. `--source cuts` scores held-out cuts from
  `skyball_gameplay.evaluation --save-predictions` output.
  `scripts/render_missed_event_gifs.py` renders one 1080p GIF per missed event.

## Reference result (full10, 8,848 events)

| Model | Missed @0.5, no pad | Dead time rejected |
|---|---|---|
| Rally net, epoch 2 | 4 (0.05%) | 87.3% |
| Prod gameplay v8 | 0 | 44.8% |

Remaining misses are single-serve rallies (aces and service errors) lasting 1-2 s,
often with the server at the frame edge. Oversampling short rallies and labelling
the 4 s before each serve as rally were tried (2026-11-06) and did not beat
epoch 2 on full10.

This is a research model. It is not integrated in the production pipeline.
