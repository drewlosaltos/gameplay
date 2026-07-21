#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current best SkyBall gameplay classifier:
# ResNet-50 encoder, four 1 Hz frames per window, average pooled embeddings,
# binary classification head, configurable square/rectangular input transform,
# Short step-LR schedule for the current SkyBall gameplay classifier:
# 2 epochs at base LR, then 10x lower LR for 2 more epochs.

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
COHORT_SUMMARIZER=${COHORT_SUMMARIZER:-${REPO_ROOT}/scripts/summarize_gameplay_validation_cohorts.py}
DSET_DIR=${DSET_DIR:-/mnt/t/data/vball/skyball/gameplay/v1}
RUN_ROOT=${RUN_ROOT:-/mnt/t/output/trn/skyball_gameplay}
DATASET_VERSION=${DATASET_VERSION:-$(basename "${DSET_DIR}")}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-224}
IMAGE_WIDTH=${IMAGE_WIDTH:-224}
EPOCHS=${EPOCHS:-4}
LR_SCHEDULER=${LR_SCHEDULER:-step}
LR_STEP_SIZE=${LR_STEP_SIZE:-2}
LR_GAMMA=${LR_GAMMA:-0.1}
CROP_SCALE_MIN=${CROP_SCALE_MIN:-0.65}
COLOR_JITTER=${COLOR_JITTER:-0.15}
AUG_POLICY=${AUG_POLICY:-none}
AUTO_AUGMENT_POLICY=${AUTO_AUGMENT_POLICY:-imagenet}
RAND_AUGMENT_OPS=${RAND_AUGMENT_OPS:-2}
RAND_AUGMENT_MAGNITUDE=${RAND_AUGMENT_MAGNITUDE:-9}
AUGMIX_SEVERITY=${AUGMIX_SEVERITY:-3}
RANDOM_ERASING=${RANDOM_ERASING:-0.1}
TEMPORAL_DROPOUT=${TEMPORAL_DROPOUT:-0.0}
TEMPORAL_SHARED_AUGMENT=${TEMPORAL_SHARED_AUGMENT:-0}
LABEL_MODE=${LABEL_MODE:-binary}
STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
if [[ "${LR_SCHEDULER}" == "cosine" ]]; then
  SCHEDULE_TAG="cos${EPOCHS}"
else
  SCHEDULE_TAG="step${EPOCHS}s${LR_STEP_SIZE}"
fi
DROP_SUFFIX=""
if [[ "${TEMPORAL_DROPOUT}" != "0" && "${TEMPORAL_DROPOUT}" != "0.0" ]]; then
  DROP_SUFFIX="_drop${TEMPORAL_DROPOUT//./}"
fi
RUN_NAME=${RUN_NAME:-${DATASET_VERSION}_resnet50_f4_avg_h${IMAGE_HEIGHT}w${IMAGE_WIDTH}_${SCHEDULE_TAG}_pre_aug_${STAMP}}

GPUS=${GPUS:-0,1}
PORT=${PORT:-1234}
CONDA_ENV=${CONDA_ENV:-pt5090new}
WORKERS=${WORKERS:-4}
TEMPORAL_BATCH_SIZE=${TEMPORAL_BATCH_SIZE:-64}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-128}
SAVE_PREDICTIONS=${SAVE_PREDICTIONS:-1}
CHECK_GPUS=${CHECK_GPUS:-1}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -d "${DSET_DIR}" ]]; then
  echo "Missing DSET_DIR: ${DSET_DIR}" >&2
  exit 2
fi

if [[ "${CHECK_GPUS}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

OUT_DIR="${RUN_ROOT}/${RUN_NAME}"
TAG="resnet50_f4_avg_${LABEL_MODE}_h${IMAGE_HEIGHT}w${IMAGE_WIDTH}_${SCHEDULE_TAG}_pre${DROP_SUFFIX}"
CHECKPOINT="${OUT_DIR}/best_${TAG}.pth.tar"
EVAL_DIR="${OUT_DIR}/eval_${TAG}"

echo "Training SkyBall gameplay temporal model"
echo "  dataset: ${DSET_DIR}"
echo "  output:  ${OUT_DIR}"
echo "  trainer: ${REPO_ROOT}/skyball_gameplay"
echo "  image:   ${IMAGE_HEIGHT}x${IMAGE_WIDTH}"
echo "  labels:  ${LABEL_MODE}"
echo "  sched:   ${LR_SCHEDULER} epochs=${EPOCHS} step_size=${LR_STEP_SIZE} gamma=${LR_GAMMA}"
echo "  aug:     crop_scale_min=${CROP_SCALE_MIN} color_jitter=${COLOR_JITTER} random_erasing=${RANDOM_ERASING}"
echo "  policy:  aug_policy=${AUG_POLICY} auto=${AUTO_AUGMENT_POLICY} rand_ops=${RAND_AUGMENT_OPS} rand_mag=${RAND_AUGMENT_MAGNITUDE} augmix_severity=${AUGMIX_SEVERITY}"
echo "  temporal: dropout=${TEMPORAL_DROPOUT} shared_augment=${TEMPORAL_SHARED_AUGMENT}"
echo "  gpus:    ${GPUS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1; command not launched."
  exit 0
fi

cd "${REPO_ROOT}"
TRAIN_ARGS=(
  -a resnet50
  --pretrained
  --epochs "${EPOCHS}"
  --lr-scheduler "${LR_SCHEDULER}"
  --lr-step-size "${LR_STEP_SIZE}"
  --lr-gamma "${LR_GAMMA}"
  --dist-url "tcp://127.0.0.1:${PORT}"
  --dist-backend nccl
  --multiprocessing-distributed
  --world-size 1
  --rank 0
  -j "${WORKERS}"
  -b "${TEMPORAL_BATCH_SIZE}"
  --input-frames 4
  --label-mode "${LABEL_MODE}"
  --image-size "${IMAGE_HEIGHT}" "${IMAGE_WIDTH}"
  --crop-scale-min "${CROP_SCALE_MIN}"
  --color-jitter "${COLOR_JITTER}"
  --aug-policy "${AUG_POLICY}"
  --auto-augment-policy "${AUTO_AUGMENT_POLICY}"
  --rand-augment-ops "${RAND_AUGMENT_OPS}"
  --rand-augment-magnitude "${RAND_AUGMENT_MAGNITUDE}"
  --augmix-severity "${AUGMIX_SEVERITY}"
  --random-erasing "${RANDOM_ERASING}"
  --temporal-dropout "${TEMPORAL_DROPOUT}"
  --output-dir "${OUT_DIR}"
  --run-tag "${TAG}"
  "${DSET_DIR}"
)
if [[ "${TEMPORAL_SHARED_AUGMENT}" == "1" ]]; then
  TRAIN_ARGS+=(--temporal-shared-augment)
fi
CUDA_VISIBLE_DEVICES="${GPUS}" conda run --no-capture-output -n "${CONDA_ENV}" \
  python -m skyball_gameplay.training "${TRAIN_ARGS[@]}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Expected checkpoint was not written: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${SAVE_PREDICTIONS}" == "1" ]]; then
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" conda run --no-capture-output -n "${CONDA_ENV}" \
    python -m skyball_gameplay.evaluation \
      --data "${DSET_DIR}" \
      --checkpoint "${CHECKPOINT}" \
      --output-dir "${EVAL_DIR}" \
      --eval-unit native \
      --save-predictions \
      -b "${EVAL_BATCH_SIZE}" \
      -j "${WORKERS}"

  if grep -q '"validation_cohort": "v5_expanded"' "${DSET_DIR}/metadata/windows.jsonl"; then
    conda run --no-capture-output -n "${CONDA_ENV}" \
      python "${COHORT_SUMMARIZER}" \
        --dataset-root "${DSET_DIR}" \
        --predictions "${EVAL_DIR}/predictions.csv" \
        --output "${EVAL_DIR}/validation_cohort_metrics.json"
  fi
fi

echo "Done."
echo "  checkpoint: ${CHECKPOINT}"
echo "  metrics:    ${EVAL_DIR}/metrics.json"
echo "  registry:   update ${REPO_ROOT}/model_registry.yaml if this run improves the leaderboard"
