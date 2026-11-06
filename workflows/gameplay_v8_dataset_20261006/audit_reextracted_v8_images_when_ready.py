#!/usr/bin/env python3
"""Fully decode every r2 JPEG from the 11 sources touched by early raw reads."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

ROOT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r2")
DATASET = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")


def write(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def status(stage: str, **details: object) -> None:
    write(ROOT / "suspect_image_audit_status.json", {
        "stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **details,
    })


def main() -> None:
    clone = json.loads((ROOT / "r2_clone_complete.json").read_text())
    suspect = clone["suspect_sources"]
    if len(suspect) != 11:
        raise RuntimeError("Expected 11 early-read sources")
    status("waiting_for_suspect_extractions")
    while True:
        extraction = ROOT / "extraction_status.json"
        if extraction.exists():
            state = json.loads(extraction.read_text())
            if state["status"] == "finished_with_failures":
                raise RuntimeError("Extraction finished with failed source videos")
        if all((WORK / "video_checks" / f"{video_id}.json").is_file() for video_id in suspect):
            break
        time.sleep(30)
    total = 0
    per_video = {}
    for video_id in suspect:
        check = json.loads((WORK / "video_checks" / f"{video_id}.json").read_text())
        if check["status"] != "complete":
            raise RuntimeError(f"Incomplete extraction check for {video_id}")
        dropped = set(check["dropped_audio_tail_windows"])
        paths = set()
        with (WORK / "windows_by_video" / f"{video_id}.jsonl").open() as handle:
            for line in handle:
                row = json.loads(line)
                if row["window_id"] not in dropped:
                    paths.update(row["frame_paths"])
        if len(paths) != check["new_images"]:
            raise RuntimeError(f"Expected freshly decoded JPEGs only for {video_id}: "
                               f"{len(paths)} paths vs {check['new_images']} new images")
        for relative in sorted(paths):
            path = DATASET / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image.load()
                if image.format != "JPEG" or max(image.size) != 1024:
                    raise RuntimeError(f"Unexpected JPEG format or dimensions: {path}, {image.format}, {image.size}")
        per_video[video_id] = {"decoded_images": len(paths), "source_check": str(WORK / "video_checks" / f"{video_id}.json")}
        total += len(paths)
        status("running", complete_sources=len(per_video), total_sources=len(suspect), decoded_images=total)
    result = {
        "status": "complete", "dataset": str(DATASET), "suspect_sources": len(suspect),
        "fully_decoded_jpegs": total, "per_video": per_video,
    }
    write(ROOT / "suspect_image_audit.json", result)
    status("complete", result=str(ROOT / "suspect_image_audit.json"), decoded_images=total)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
