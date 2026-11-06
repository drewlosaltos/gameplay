#!/usr/bin/env bash
set -u
report_dir=/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006
source_remote='googdrive,root_folder_id=1WN1u5qRo2-hI_iMkj1J3jHiRq73Ov_ch:'
dest_root=/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user
rclone check "$source_remote" "$dest_root" \
  --files-from "$report_dir/selected_files.txt" --checksum \
  --differ "$report_dir/videos_differ.txt" \
  --missing-on-dst "$report_dir/videos_missing_on_dst.txt" \
  --missing-on-src "$report_dir/videos_missing_on_src.txt" \
  --error "$report_dir/videos_error.txt" \
  --combined "$report_dir/videos_combined.txt" \
  --log-file "$report_dir/precheck.log" --log-level INFO
check_status=$?
printf '%s\n' "$check_status" > "$report_dir/precheck.exitcode"
if [[ -s "$report_dir/videos_error.txt" || "$check_status" -gt 1 ]]; then
  printf 'Precheck errored; copy not started.\n' >> "$report_dir/sync.log"
  exit 2
fi
if [[ -s "$report_dir/videos_differ.txt" ]]; then
  printf 'Existing files differ; copy still limited to missing paths.\n' >> "$report_dir/sync.log"
fi
rclone copy "$source_remote" "$dest_root" \
  --files-from "$report_dir/videos_missing_on_dst.txt" \
  --ignore-existing --checksum --transfers 4 --checkers 8 --stats 60s \
  --log-file "$report_dir/copy.log" --log-level INFO
copy_status=$?
printf '%s\n' "$copy_status" > "$report_dir/copy.exitcode"
if [[ "$copy_status" -ne 0 ]]; then
  printf 'Copy failed; see copy.log.\n' >> "$report_dir/sync.log"
  exit "$copy_status"
fi
rclone check "$source_remote" "$dest_root" \
  --files-from "$report_dir/selected_files.txt" --checksum \
  --differ "$report_dir/final_differ.txt" \
  --missing-on-dst "$report_dir/final_missing_on_dst.txt" \
  --missing-on-src "$report_dir/final_missing_on_src.txt" \
  --error "$report_dir/final_error.txt" \
  --combined "$report_dir/final_combined.txt" \
  --log-file "$report_dir/final_check.log" --log-level INFO
final_status=$?
printf '%s\n' "$final_status" > "$report_dir/final_check.exitcode"
exit "$final_status"
