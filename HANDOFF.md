## Production promotion — 2026-10-07

The user approved the V8 single-stage Qwen candidate as production, the exact
FP8 recipe as the indoor default, and clean val48 as default validation (no
val20/val50 reevaluation). See `workflows/qwen_v8_production_20261007.md`,
`config/indoor_canonical.json`, and `scripts/train_qwen_canonical.py`.
Model: `v8_val48_qwen672x384_single_stage_fp8_20261006_r3` (epoch 3).
Pipeline release: `skyball-prod-20261007-r1`; threshold 0.9, pad 5s/side, gap 0.
Publication/evidence: `/mnt/t/output/skyball_gameplay/qwen_v8_publication_20261007_r1`.
Reproduce it with `REPRODUCE.md`. Earlier handoff notes (v0 dataset extraction) are in
`HISTORY.md`.
