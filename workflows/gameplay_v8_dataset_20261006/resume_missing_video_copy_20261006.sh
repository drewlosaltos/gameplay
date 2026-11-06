#!/usr/bin/env bash
set -euo pipefail
report_dir=/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006
source_remote='googdrive,root_folder_id=1WN1u5qRo2-hI_iMkj1J3jHiRq73Ov_ch:'
dest_root=/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user
rclone copy "$source_remote" "$dest_root" \
  --files-from "$report_dir/videos_missing_on_dst.txt" \
  --ignore-existing --checksum --transfers 4 --checkers 8 --stats 60s \
  --log-file "$report_dir/copy_resume_20261006.log" --log-level INFO
