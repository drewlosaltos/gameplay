#!/usr/bin/env bash
set -euo pipefail

# Four-frame SigLIP-Base gameplay experiment. The augmentation and short
# step schedule match the v5 ResNet-50 research SOTA; AdamW and the conservative
# encoder learning rate come from the successful jersey SigLIP recipe.

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
COHORT_SUMMARIZER=${COHORT_SUMMARIZER:-${REPO_ROOT}/scripts/summarize_gameplay_validation_cohorts.py}
DSET_DIR=${DSET_DIR:-/mnt/t/data/vball/skyball/gameplay/v5}
RUN_ROOT=${RUN_ROOT:-/mnt/t/output/trn/skyball_gameplay}
TIMM_MODEL=${TIMM_MODEL:-vit_base_patch16_siglip_224.v2_webli}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-224}
IMAGE_WIDTH=${IMAGE_WIDTH:-224}
RUN_NAME=${RUN_NAME:-v5_siglip_base_f4_avg_h${IMAGE_HEIGHT}w${IMAGE_WIDTH}_step4s2_lr1e4_blr5e6_20260719}
OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
TAG=${TAG:-siglip_base_f4_avg_binary_h${IMAGE_HEIGHT}w${IMAGE_WIDTH}_step4s2_pre_drop010}
CHECKPOINT="${OUT_DIR}/best_${TAG}.pth.tar"
EVAL_DIR="${OUT_DIR}/eval_${TAG}"

GPUS=${GPUS:-0,1,2,3}
PORT=${PORT:-1241}
CONDA_ENV=${CONDA_ENV:-pt5090new}
WORKERS=${WORKERS:-16}
TEMPORAL_BATCH_SIZE=${TEMPORAL_BATCH_SIZE:-128}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
CHECK_GPUS=${CHECK_GPUS:-1}
DRY_RUN=${DRY_RUN:-0}

echo "Training SkyBall SigLIP temporal model"
echo "  dataset: ${DSET_DIR}"
echo "  output:  ${OUT_DIR}"
echo "  trainer: ${REPO_ROOT}/skyball_gameplay"
echo "  model:   ${TIMM_MODEL}"
echo "  image:   ${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
echo "  gpus:    ${GPUS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; command not launched."
  exit 0
fi

if [[ "${CHECK_GPUS}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

mkdir -p "${OUT_DIR}"
if [[ -e "${OUT_DIR}/last_${TAG}.pth.tar" || -e "${CHECKPOINT}" ]]; then
  echo "Refusing to overwrite an existing SigLIP run in ${OUT_DIR}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${GPUS}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m skyball_gameplay.training \
    --timm-model "${TIMM_MODEL}" \
    --pretrained \
    --optimizer adamw \
    --lr 1e-4 \
    --backbone-lr-multiplier 0.05 \
    --weight-decay 1e-4 \
    --epochs 4 \
    --lr-scheduler step \
    --lr-step-size 2 \
    --lr-gamma 0.1 \
    --amp \
    --dist-url "tcp://127.0.0.1:${PORT}" \
    --dist-backend nccl \
    --multiprocessing-distributed \
    --world-size 1 \
    --rank 0 \
    -j "${WORKERS}" \
    -b "${TEMPORAL_BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --input-frames 4 \
    --label-mode binary \
    --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}" \
    --crop-scale-min 0.65 \
    --color-jitter 0.10 \
    --aug-policy randaugment \
    --rand-augment-ops 2 \
    --rand-augment-magnitude 5 \
    --random-erasing 0.1 \
    --temporal-dropout 0.1 \
    --temporal-shared-augment \
    --output-dir "${OUT_DIR}" \
    --run-tag "${TAG}" \
    "${DSET_DIR}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected checkpoint was not written: ${CHECKPOINT}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPUS%%,*}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m skyball_gameplay.evaluation \
    --data "${DSET_DIR}" \
    --checkpoint "${CHECKPOINT}" \
    --output-dir "${EVAL_DIR}" \
    --eval-unit native \
    --save-predictions \
    -b "${EVAL_BATCH_SIZE}" \
    -j "${WORKERS}"

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${COHORT_SUMMARIZER}" \
    --dataset-root "${DSET_DIR}" \
    --predictions "${EVAL_DIR}/predictions.csv" \
    --output "${EVAL_DIR}/validation_cohort_metrics.json"

echo "completed=true"
echo "checkpoint=${CHECKPOINT}"
echo "metrics=${EVAL_DIR}/metrics.json"
