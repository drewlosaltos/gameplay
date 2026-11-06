#!/usr/bin/env bash
set -euo pipefail
cd /home/atao/vsdevel/skyball/gameplay
conda run --no-capture-output -n pt5090new python scripts/make_review_clips.py \
  --mode dense \
  --workbook /mnt/t/output/skyball_gameplay/v8_workflow_20261006/review_labels.xlsx \
  --raw-video-root /mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user \
  --output-root /mnt/t/output/skyball_gameplay/v8_workflow_20261006 \
  --run-name review_clips_v8_binary_balanced_r2 \
  --num-videos 4 --samples-per-video 12 --binary-balanced \
  --clip-seconds 4 --boundary-margin-sec 30 --seed 20261006 \
  --width 960 --encoder h264_nvenc \
  > /mnt/t/output/skyball_gameplay/v8_workflow_20261006/review_clips_balanced.log 2>&1
