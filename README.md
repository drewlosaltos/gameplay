# SkyBall gameplay classification

Decides whether short windows of SkyBall volleyball video are live gameplay. The
pipeline uses it to skip dead time. Indoor and beach models both train from `main`.

**Production (indoor):** `v8_val48_qwen672x384_single_stage_fp8_20261006_r3`.
- Model: four 1 Hz frames through a Qwen3.6 vision encoder at 672x384 (gray128
  letterbox), with mean pooling over time and a binary MLP head.
- Training: one stage from pretrained Qwen with a fresh head, no gameplay
  checkpoint. Four epochs on four GPUs, global batch 16, AdamW (head 1e-4,
  encoder 3e-6), step decay x0.1 every 2 epochs. TorchAO FP8 + `torch.compile`
  under BF16, with FP32 validation.
- Result: epoch 3 kept, with clean val48 F1 **0.961224** on 16,281 windows
  (previous production: 0.946217). In the pipeline it runs at threshold 0.9, with
  5 s padding per side and gap 0.
- **To reproduce it, or to train a new indoor model, start at [`REPRODUCE.md`](REPRODUCE.md)**:
  `python scripts/reproduce.py recipes/<model_id>.json`.

**Beach:** `v7_beach_siglip512_prod_finetune_gpu0_20260723`, a SigLIP-Base fine-tune.
See [`workflows/beach_training.md`](workflows/beach_training.md).

| Where | What |
|---|---|
| `recipes/`, `scripts/reproduce.py` | Reproduction recipes (schema `skyball_repro_recipe/v1`) and the one-command launcher |
| `config/indoor_canonical.json`, `scripts/train_qwen_canonical.py` | Canonical indoor hyperparameters and launcher |
| `environment.yml`, `requirements-lock.txt` | The gameplay training env `skyball-gameplay-train` (lock) |
| `validation/` | Frozen splits (`gameplay_indoor_v8_val48_clean_v2.json`) and gate contracts |
| `model_registry.yaml` | Current models with their metrics and evidence |
| `workflows/` | Dataset builds (`gameplay_v8_dataset.md`), full10 gate (`gameplay_gate_validation.md`), prod promotion (`qwen_v8_production_20261007.md`), rally net |
| `HISTORY.md`, `docs/history/` | Superseded recipes (SigLIP, two-stage, v5 Qwen), old handoff notes and registry entries |

## Two required validation sources

1. **Classification:** F1, precision, recall and confusion at threshold 0.5 on
   clean val48, comparing production/candidate on identical samples. Report the selected epoch, not just the last epoch.
2. **Ten full matches:** confirmed events excluded by the gate, and extra
   footage admitted as a percentage of the entire match. Extra footage is
   outside the union of +/-3-second windows around confirmed GT events.
   Pool seconds across matches because FPS differs. Prioritize zero excluded
   events, then compare extra admitted footage.

Canonical inputs:
`/mnt/t/output/skyball_pipeline/evaluation_full10_h_primary30fps_20260806_v1`.
Preserve its two 30-fps transcodes and remapped GT. Reuse only verified
production artifacts. The scorer uses production RGB preprocessing; the
analyzer reconstructs saved production admission exactly. State threshold,
padding per side, merge gap and runtime. Compare identical settings separately
from selected policies. These ten matches are a second validation source, not
an independent test estimate after model/policy selection.

Current production policy: threshold **0.9**, **5 seconds per side**, gap 0.
This is separate from the 0.5 classifier threshold and travels with the model.
Production requires portable/runtime qualification and pipeline acceptance;
a high classification F1 alone is insufficient.

See [training progression](workflows/human_gameplay_finetune_v0.md),
[full-video validation](workflows/gameplay_gate_validation.md), and contracts
in `validation/`. Published bundles must include dataset locations/hashes,
split membership, training lineage and commands, selected epochs, both
validation sources, and limitations. Model-assisted labels are not verified
untouched LLM output.
