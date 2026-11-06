#!/usr/bin/env python3
"""Find extraction attempts that began before rclone finished the raw source."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1")
SYNC = Path("/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006")
OUT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006/early_raw_read_audit.json")
LOCAL_TZ = timezone(-timedelta(hours=7))


def main() -> None:
    completed: dict[str, datetime] = {}
    for path in (SYNC / "copy.log", SYNC / "copy_resume_20261006.log"):
        for line in path.read_text(errors="replace").splitlines():
            match = re.match(
                r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) INFO  : ([^:]+\.mp4): Multi-thread Copied \(new\)",
                line,
            )
            if match:
                finished = datetime.strptime(match[1], "%Y/%m/%d %H:%M:%S")
                completed[Path(match[2]).stem] = finished.replace(tzinfo=LOCAL_TZ).astimezone(timezone.utc)
    rows = []
    for path in sorted((WORK / "video_logs").glob("*.log")):
        attempts = [datetime.fromisoformat(match)
                    for match in re.findall(r"^START (\S+)", path.read_text(errors="replace"), re.MULTILINE)]
        if not attempts:
            continue
        first = min(attempts)
        finished = completed.get(path.stem)
        check_path = WORK / "video_checks" / f"{path.stem}.json"
        check = json.loads(check_path.read_text()) if check_path.exists() else {}
        row = {"video_id": path.stem, "first_extraction_start_utc": first.isoformat(),
               "copy_completion_utc": finished.isoformat() if finished else None,
               "started_before_copy_complete": finished is not None and first < finished,
               "check_complete": check.get("status") == "complete",
               "new_images": check.get("new_images"), "strategy": check.get("strategy")}
        rows.append(row)
    early = [row for row in rows if row["started_before_copy_complete"]]
    suspect = {row["video_id"] for row in early
               if not row["check_complete"] or (row["new_images"] or 0) > 0}
    # A killed extractor can leave images without a check even if it started
    # after the copy completed; those images also need fresh extraction.
    for path in (WORK / "video_logs").glob("*.log"):
        if (WORK / "video_checks" / f"{path.stem}.json").exists():
            continue
        suspect.add(path.stem)
    result = {"status": "audited", "attempts": len(rows),
              "early_start_count": len(early), "early_starts": early,
              "suspect_video_ids": sorted(suspect),
              "suspect_count": len(suspect),
              "policy": "Preserve r1 untouched; create r2 from non-suspect hardlinks and re-extract suspect sources from completed raw files."}
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"attempts": len(rows), "early_starts": len(early),
                      "suspect": len(suspect), "report": str(OUT)}))


if __name__ == "__main__":
    main()
