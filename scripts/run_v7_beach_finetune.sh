#!/usr/bin/env bash
set -euo pipefail

# Import the 47 beach-only additions in v7, build a source-held-out split,
# evaluate the production model, and fine-tune that checkpoint on GPU 0.
# Historical end-to-end import: requires a new dataset directory and explicit
# authorization for non-blocking review. To train an existing approved beach
# dataset, use scripts/train_siglip_temporal.sh as documented in README.md.

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
WORKBOOK=${WORKBOOK:-/mnt/t/data/vball/skyball/gameplay/ground_truth/v7.xlsx}
RAW_ROOT=${RAW_ROOT:-/mnt/t/data/vball/skyball/raw_video}
DATASET_ROOT=${DATASET_ROOT:-/mnt/t/data/vball/skyball/gameplay/v7}
POPULATION_MANIFEST=${POPULATION_MANIFEST:-${REPO_ROOT}/validation/gameplay_v7_beach_population_v1.json}
VAL_MANIFEST=${VAL_MANIFEST:-${REPO_ROOT}/validation/gameplay_v7_beach_val_v1.json}
SYNC_ROOT=${SYNC_ROOT:-/mnt/t/data/vball/skyball/gameplay/v7_raw_sync}
FILES_FROM=${FILES_FROM:-${SYNC_ROOT}/rclone_raw_videos_files_from.txt}
RUN_ROOT=${RUN_ROOT:-/mnt/t/output/skyball_gameplay/v7_beach_build_finetune_20260723}
RECON_ROOT=${RECON_ROOT:-${RUN_ROOT}/reconciliation}
REVIEW_ROOT=${REVIEW_ROOT:-/mnt/t/output/skyball_gameplay/review_clips}
REVIEW_RUN_NAME=${REVIEW_RUN_NAME:-v7_beach_20260723}
REVIEW_DIR=${REVIEW_ROOT}/${REVIEW_RUN_NAME}
PROD_CHECKPOINT=${PROD_CHECKPOINT:-/mnt/t/output/trn/skyball_gameplay/v5_siglip_base_f4_avg_h512w512_step4s2_lr1e4_blr5e6_20260719/best_siglip_base_f4_avg_binary_h512w512_step4s2_pre_drop010.pth.tar}
TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-v7_beach_siglip512_prod_finetune_gpu0_20260723}
TRAIN_ROOT=${TRAIN_ROOT:-/mnt/t/output/trn/skyball_gameplay}
BASELINE_EVAL_DIR=${BASELINE_EVAL_DIR:-${RUN_ROOT}/prod_checkpoint_beach_val}
CONDA_ENV=${CONDA_ENV:-pt5090new}
CONTROL_LOG=${CONTROL_LOG:-${RUN_ROOT}/control.log}

mkdir -p "${RUN_ROOT}" "${RECON_ROOT}"
exec > >(tee -a "${CONTROL_LOG}") 2>&1

stage() {
  echo
  echo "[$(date -Is)] $*"
}

stage "resolving the v7 beach raw-video paths"
if [[ ! -s "${FILES_FROM}" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/build_gameplay_v0_dataset.py" sync-raw \
      --workbook "${WORKBOOK}" \
      --raw-root "${RAW_ROOT}" \
      --output-root "${DATASET_ROOT}" \
      --video-manifest "${POPULATION_MANIFEST}" \
      --dry-run \
      --skip-duration-check
fi
test "$(wc -l < "${FILES_FROM}")" -eq 47

stage "checksum-reconciling the selected raw-video population"
check_status=0
rclone check googdrive:skyball/raw_video "${RAW_ROOT}" \
  --files-from "${FILES_FROM}" \
  --one-way \
  --checksum \
  --checkers 32 \
  --differ "${RECON_ROOT}/raw_pre_differ.txt" \
  --missing-on-dst "${RECON_ROOT}/raw_pre_missing_on_dst.txt" \
  --missing-on-src "${RECON_ROOT}/raw_pre_missing_on_src.txt" \
  --error "${RECON_ROOT}/raw_pre_error.txt" \
  --combined "${RECON_ROOT}/raw_pre_combined.txt" || check_status=$?
test ! -s "${RECON_ROOT}/raw_pre_differ.txt"
test ! -s "${RECON_ROOT}/raw_pre_missing_on_src.txt"
test ! -s "${RECON_ROOT}/raw_pre_error.txt"
if [[ "${check_status}" -ne 0 && ! -s "${RECON_ROOT}/raw_pre_missing_on_dst.txt" ]]; then
  echo "Raw reconciliation failed without any missing-on-destination files" >&2
  exit "${check_status}"
fi

missing_count=$(wc -l < "${RECON_ROOT}/raw_pre_missing_on_dst.txt")
stage "copying ${missing_count} missing beach raw videos without replacing local files"
if [[ "${missing_count}" -gt 0 ]]; then
  rclone copy googdrive:skyball/raw_video "${RAW_ROOT}" \
    --files-from "${RECON_ROOT}/raw_pre_missing_on_dst.txt" \
    --ignore-existing \
    --create-empty-src-dirs \
    --transfers 4 \
    --checkers 16 \
    --stats 30s
fi

stage "verifying the complete selected raw-video population"
rclone check googdrive:skyball/raw_video "${RAW_ROOT}" \
  --files-from "${FILES_FROM}" \
  --one-way \
  --checksum \
  --checkers 32 \
  --differ "${RECON_ROOT}/raw_post_differ.txt" \
  --missing-on-dst "${RECON_ROOT}/raw_post_missing_on_dst.txt" \
  --missing-on-src "${RECON_ROOT}/raw_post_missing_on_src.txt" \
  --error "${RECON_ROOT}/raw_post_error.txt" \
  --combined "${RECON_ROOT}/raw_post_combined.txt"

stage "validating workbook timestamps against the 47 local raw videos"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/scripts/build_gameplay_v0_dataset.py" validate-raw \
    --workbook "${WORKBOOK}" \
    --raw-root "${RAW_ROOT}" \
    --metadata-dir "${RUN_ROOT}/raw_duration_validation" \
    --video-manifest "${POPULATION_MANIFEST}"

if [[ -f "${REVIEW_DIR}/summary.json" ]] \
  && [[ "$(wc -l < "${REVIEW_DIR}/review_manifest.jsonl")" -eq 48 ]] \
  && [[ "$(find "${REVIEW_DIR}/combined" -maxdepth 1 -type f -name '*_review.mp4' | wc -l)" -eq 4 ]]; then
  stage "using completed v7 beach review clips at ${REVIEW_DIR}"
elif [[ -e "${REVIEW_DIR}" ]]; then
  echo "Incomplete review output already exists: ${REVIEW_DIR}" >&2
  exit 2
else
  stage "rendering non-blocking review clips from four beach matches"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${REPO_ROOT}/scripts/make_review_clips.py" \
      --mode dense \
      --workbook "${WORKBOOK}" \
      --raw-video-root "${RAW_ROOT}" \
      --output-root "${REVIEW_ROOT}" \
      --run-name "${REVIEW_RUN_NAME}" \
      --videos \
        000632_atomic_krait \
        003318_capricious_bowerbird \
        2023-07-31-W-Montreal-Flint-Scoles-vs-Mel-Brandie-401472 \
        2024-05-31-M-StareJablonki-Diaz-Alayo-vs-Krattiger-Breer-509641 \
      --samples-per-video 12 \
      --clip-seconds 4 \
      --boundary-margin-sec 30 \
      --seed 20260723 \
      --encoder h264_nvenc
fi

if [[ -e "${DATASET_ROOT}" ]]; then
  echo "Refusing to overwrite existing v7 dataset: ${DATASET_ROOT}" >&2
  exit 2
fi

stage "extracting the beach-only v7 train/validation dataset"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/scripts/build_gameplay_v0_dataset.py" extract \
    --workbook "${WORKBOOK}" \
    --raw-root "${RAW_ROOT}" \
    --output-root "${DATASET_ROOT}" \
    --video-manifest "${POPULATION_MANIFEST}" \
    --val-video-manifest "${VAL_MANIFEST}" \
    --val-cohort-name beach_v7 \
    --gameplay-stride-sec 20 \
    --negative-stride-sec 10 \
    --train-negative-stride-sec 10 \
    --long-edge 1024 \
    --jpeg-quality 88 \
    --extract-workers 4 \
    --seed 20260723

stage "auditing image existence, timing, and source split isolation"
conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/scripts/validate_gameplay_dataset.py" \
    --dataset-root "${DATASET_ROOT}" \
    --video-manifest "${POPULATION_MANIFEST}" \
    --val-video-manifest "${VAL_MANIFEST}" \
    --output "${RUN_ROOT}/dataset_validation.json"

stage "evaluating the current production checkpoint on held-out beach validation"
mkdir -p "${BASELINE_EVAL_DIR}"
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m skyball_gameplay.evaluation \
    --data "${DATASET_ROOT}" \
    --checkpoint "${PROD_CHECKPOINT}" \
    --output-dir "${BASELINE_EVAL_DIR}" \
    --eval-unit native \
    --save-predictions \
    -b 32 \
    -j 8
conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${REPO_ROOT}/scripts/summarize_gameplay_validation_cohorts.py" \
    --dataset-root "${DATASET_ROOT}" \
    --predictions "${BASELINE_EVAL_DIR}/predictions.csv" \
    --output "${BASELINE_EVAL_DIR}/validation_cohort_metrics.json"

stage "fine-tuning the production-selected SigLIP checkpoint on GPU 0"
cd "${REPO_ROOT}"
DSET_DIR="${DATASET_ROOT}" \
RUN_ROOT="${TRAIN_ROOT}" \
RUN_NAME="${TRAIN_RUN_NAME}" \
TIMM_MODEL=vit_base_patch16_siglip_512.v2_webli \
IMAGE_HEIGHT=512 \
IMAGE_WIDTH=512 \
GPUS=0 \
PORT=1247 \
WORKERS=8 \
TEMPORAL_BATCH_SIZE=16 \
GRAD_ACCUM_STEPS=8 \
EVAL_BATCH_SIZE=32 \
INIT_CHECKPOINT="${PROD_CHECKPOINT}" \
CONDA_ENV="${CONDA_ENV}" \
  "${REPO_ROOT}/scripts/train_siglip_temporal.sh"

stage "v7 beach import, split, production fine-tune, and evaluation completed"
touch "${RUN_ROOT}/completed.sentinel"
