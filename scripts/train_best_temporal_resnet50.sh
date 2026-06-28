#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current best SkyBall gameplay classifier:
# ResNet-50 encoder, four 1 Hz frames per window, average pooled embeddings,
# binary classification head, 224x224 input transform, 10 epoch cosine LR.

IMAGENET_EXAMPLE_DIR=${IMAGENET_EXAMPLE_DIR:-/home/atao/vsdevel/public/examples/imagenet}
DSET_DIR=${DSET_DIR:-/mnt/t/data/vball/skyball/gameplay/v1}
RUN_ROOT=${RUN_ROOT:-/mnt/t/output/trn/skyball_gameplay}
DATASET_VERSION=${DATASET_VERSION:-$(basename "${DSET_DIR}")}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_NAME=${RUN_NAME:-${DATASET_VERSION}_resnet50_f4_avg_h224w224_cos10_pre_${STAMP}}

GPUS=${GPUS:-0,1}
PORT=${PORT:-1234}
CONDA_ENV=${CONDA_ENV:-pt5090new}
WORKERS=${WORKERS:-4}
TEMPORAL_BATCH_SIZE=${TEMPORAL_BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
SAVE_PREDICTIONS=${SAVE_PREDICTIONS:-1}
CHECK_GPUS=${CHECK_GPUS:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -d "${IMAGENET_EXAMPLE_DIR}" ]]; then
  echo "Missing IMAGENET_EXAMPLE_DIR: ${IMAGENET_EXAMPLE_DIR}" >&2
  exit 2
fi

if [[ ! -d "${DSET_DIR}" ]]; then
  echo "Missing DSET_DIR: ${DSET_DIR}" >&2
  exit 2
fi

if [[ "${CHECK_GPUS}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
TAG="resnet50_f4_avg_binary_h224w224_cos10_pre"
CHECKPOINT="${OUT_DIR}/best_${TAG}.pth.tar"
EVAL_DIR="${OUT_DIR}/eval_${TAG}"

echo "Training SkyBall gameplay temporal model"
echo "  dataset: ${DSET_DIR}"
echo "  output:  ${OUT_DIR}"
echo "  trainer: ${IMAGENET_EXAMPLE_DIR}"
echo "  gpus:    ${GPUS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; command not launched."
  exit 0
fi

cd "${IMAGENET_EXAMPLE_DIR}"

env \
  DSET_DIR="${DSET_DIR}" \
  RUN_ROOT="${RUN_ROOT}" \
  RUN_NAME="${RUN_NAME}" \
  GPUS="${GPUS}" \
  PORT="${PORT}" \
  CONDA_ENV="${CONDA_ENV}" \
  EPOCHS=10 \
  PRETRAINED=1 \
  IMAGE_HEIGHT=224 \
  IMAGE_WIDTH=224 \
  TEMPORAL_BATCH_SIZE="${TEMPORAL_BATCH_SIZE}" \
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
  WORKERS="${WORKERS}" \
  EVAL_AFTER=1 \
  bash dist_train_imgnet.sh temporal

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected checkpoint was not written: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python evaluate_gameplay.py \
      --data "${DSET_DIR}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${EVAL_DIR}" \
      --eval-unit native \
      --save-predictions \
      -b "${EVAL_BATCH_SIZE}" \
      -j "${WORKERS}"
fi

echo "Done."
echo "  checkpoint: ${CHECKPOINT}"
echo "  metrics:    ${EVAL_DIR}/metrics.json"
echo "  registry:   update /home/atao/vsdevel/skyball/gameplay/model_registry.yaml if this run improves the leaderboard"
