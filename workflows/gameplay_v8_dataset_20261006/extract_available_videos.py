#!/usr/bin/env python3
"""Extract ready v8 raw videos as the missing-only Drive copy advances."""
import concurrent.futures
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r2")
ROOT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
PYTHON = "/home/atao/.miniconda/envs/pt5090new/bin/python"
EXTRACT = "/home/atao/vsdevel/skyball/gameplay/scripts/extract_indoor_v8_video.py"
LISTING = Path("/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/shared_video_listing.json")
REVIEW = ROOT/"review_clips_v8_binary_balanced_r2/summary.json"
MAX_WORKERS = 8

def write_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)

def run_video(video_id):
    logs = WORK/"video_logs"
    logs.mkdir(exist_ok=True)
    log_path = logs/f"{video_id}.log"
    command = [PYTHON,EXTRACT,video_id,"--work",str(WORK),"--gpu","3"]
    with log_path.open("a") as handle:
        handle.write(f"\nSTART {datetime.now(timezone.utc).isoformat()} {command}\n")
        handle.flush()
        result = subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT,
                                cwd="/home/atao/vsdevel/skyball/gameplay",
                                env=dict(os.environ,PYTHONPATH="/home/atao/vsdevel/skyball/gameplay"))
        handle.write(f"EXIT {result.returncode} {datetime.now(timezone.utc).isoformat()}\n")
    return result.returncode

def main():
    plan = json.loads((WORK/"plan.json").read_text())
    raw = Path(plan["raw_root"])
    overrides = json.loads((WORK / "raw_overrides.json").read_text())
    remote = {row["Name"]:(int(row["Size"]),
              datetime.fromisoformat(row["ModTime"].replace("Z", "+00:00")).timestamp())
              for row in json.loads(LISTING.read_text())}
    ids = sorted(path.stem for path in (WORK/"windows_by_video").glob("*.jsonl"))
    assert len(ids) == plan["train_sources"] + plan["new_val30_sources"]
    while not (WORK/"base_reuse_complete.json").exists() or not REVIEW.exists():
        write_json(ROOT/"extraction_status.json",{
            "status":"waiting_for_frame_reuse_and_balanced_review",
            "reuse_complete":(WORK/"base_reuse_complete.json").exists(),
            "review_complete":REVIEW.exists(),"total_videos":len(ids),
            "updated_at":datetime.now(timezone.utc).isoformat()})
        time.sleep(15)
    attempts = {video_id:0 for video_id in ids}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        active = {}
        while True:
            done = {path.stem for path in (WORK/"video_checks").glob("*.json")} if (WORK/"video_checks").exists() else set()
            for future in list(active):
                if not future.done():
                    continue
                video_id = active.pop(future)
                code = future.result()
                if code:
                    print(json.dumps({"event":"video_failed","video_id":video_id,
                                      "returncode":code,"attempt":attempts[video_id]}),flush=True)
            if len(done) == len(ids):
                write_json(ROOT/"extraction_status.json",{
                    "status":"complete","completed":len(done),"total_videos":len(ids),
                    "elapsed_sec":round(time.monotonic()-started,1),
                    "updated_at":datetime.now(timezone.utc).isoformat()})
                break
            active_ids = set(active.values())
            ready = []
            for video_id in ids:
                if video_id in done or video_id in active_ids or attempts[video_id] >= 2:
                    continue
                path = Path(overrides.get(video_id, str(raw/f"{video_id}.mp4")))
                if (path.is_file() and path.stat().st_size == remote[path.name][0]
                        and abs(path.stat().st_mtime - remote[path.name][1]) <= 1.0):
                    ready.append(video_id)
            for video_id in ready[:MAX_WORKERS-len(active)]:
                attempts[video_id] += 1
                active[pool.submit(run_video,video_id)] = video_id
            failed = [video_id for video_id in ids if attempts[video_id] >= 2
                      and video_id not in done and video_id not in active_ids]
            status = {
                "status":"running",
                "completed":len(done),"total_videos":len(ids),
                "ready_waiting":max(0,len(ready)-(MAX_WORKERS-len(active))),
                "active":list(active.values()),"failed_after_retry":failed,
                "elapsed_sec":round(time.monotonic()-started,1),
                "updated_at":datetime.now(timezone.utc).isoformat(),
            }
            write_json(ROOT/"extraction_status.json",status)
            if failed and len(done)+len(failed) == len(ids) and not active:
                status["status"]="finished_with_failures"
                write_json(ROOT/"extraction_status.json",status)
                break
            time.sleep(10)

if __name__=="__main__":
    main()
