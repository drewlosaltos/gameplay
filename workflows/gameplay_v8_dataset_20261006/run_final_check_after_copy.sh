#!/usr/bin/env bash
set -u
report_dir=/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006
source_remote='googdrive,root_folder_id=1WN1u5qRo2-hI_iMkj1J3jHiRq73Ov_ch:'
dest_root=/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user
while pgrep -f '^rclone copy googdrive,root_folder_id=1WN1u5qRo2-hI_iMkj1J3jHiRq73Ov_ch:' >/dev/null; do
  sleep 30
done
rclone check "$source_remote" "$dest_root" \
  --files-from "$report_dir/selected_files.txt" --checksum \
  --differ "$report_dir/final_differ.txt" \
  --missing-on-dst "$report_dir/final_missing_on_dst.txt" \
  --missing-on-src "$report_dir/final_missing_on_src.txt" \
  --error "$report_dir/final_error.txt" \
  --combined "$report_dir/final_combined.txt" \
  --log-file "$report_dir/final_check.log" --log-level INFO
status=$?
printf '%s\n' "$status" > "$report_dir/final_check.exitcode"
exit "$status"
