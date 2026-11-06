#!/usr/bin/env python3
"""Publish v8 manifests after all raw-video extraction checks succeed."""
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r2")
DATASET = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")
SCRIPT = Path("/home/atao/vsdevel/skyball/gameplay/scripts/finalize_indoor_v8_dataset.py")

def main():
    while True:
        path = ROOT/"extraction_status.json"
        if path.exists():
            status = json.loads(path.read_text())
            if status["status"] == "complete":
                break
            if status["status"] == "finished_with_failures":
                raise RuntimeError("Extraction has failed videos; inspect extraction_status.json")
        time.sleep(30)
    while True:
        path = ROOT/"composite_raw_check_status.json"
        if path.exists():
            check = json.loads(path.read_text())
            if check["stage"] == "failed":
                raise RuntimeError("Selected raw-video checksum audit failed; inspect composite_raw_check_status.json")
            if check["stage"] == "complete":
                result_path = Path(check["result"])
                result = json.loads(result_path.read_text())
                if (result.get("status") != "pass" or result.get("selected_total") != 1081
                        or result["main"]["exitcode"] != 0 or result["recovered"]["exitcode"] != 0):
                    raise RuntimeError("Selected raw-video checksum evidence is incomplete")
                break
        time.sleep(30)
    with (ROOT/"finalize.log").open("x") as handle:
        result = subprocess.run(
            ["/home/atao/.miniconda/envs/pt5090new/bin/python",str(SCRIPT),"--work",str(WORK)],
            stdout=handle,stderr=subprocess.STDOUT,
            cwd="/home/atao/vsdevel/skyball/gameplay")
    if result.returncode:
        raise RuntimeError(f"V8 finalizer failed with exit {result.returncode}")
    summary = json.loads((DATASET/"metadata/summary.json").read_text())
    if summary["source_video_counts"] != {"train":1031,"val20":20,"new30":30}:
        raise RuntimeError("Unexpected v8 source counts")
    ready = {"status":"ready","dataset":str(DATASET),
             "window_count":summary["window_count"],
             "image_count":summary["image_count"],
             "metadata_summary":str(DATASET/"metadata/summary.json"),
             "updated_at":datetime.now(timezone.utc).isoformat()}
    (ROOT/"dataset_ready.json").write_text(json.dumps(ready,indent=2)+"\n")
    print(json.dumps(ready),flush=True)

if __name__=="__main__":
    main()
