#!/usr/bin/env python3
"""Checksum the selected v8 raw population across original and recovered roots."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
SYNC = Path("/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006")
OUT = SYNC / "composite_check_r1"
SOURCE = "googdrive,root_folder_id=1WN1u5qRo2-hI_iMkj1J3jHiRq73Ov_ch:"
MAIN = Path("/mnt/t/data/vball/skyball/raw_video/indoor_endline/skyball-user")
RECOVERED = Path("/mnt/t/data/vball/skyball/raw_video/indoor_endline/v8_recovered_20261006_r1")


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def status(stage: str, **details: object) -> None:
    write(ROOT / "composite_raw_check_status.json", {"stage": stage,
          "updated_at": datetime.now(timezone.utc).isoformat(), **details})


def active(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def check(label: str, destination: Path, listing: Path) -> dict:
    directory = OUT / label
    directory.mkdir()
    command = ["rclone", "check", SOURCE, str(destination), "--files-from", str(listing),
               "--checksum", "--checkers", "8", "--stats", "60s",
               "--differ", str(directory / "differ.txt"),
               "--missing-on-dst", str(directory / "missing_on_dst.txt"),
               "--missing-on-src", str(directory / "missing_on_src.txt"),
               "--error", str(directory / "error.txt"),
               "--combined", str(directory / "combined.txt")]
    with (directory / "check.log").open("x") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    counts = {key: len((directory / f"{key}.txt").read_text().splitlines())
              for key in ("differ", "missing_on_dst", "missing_on_src", "error", "combined")}
    return {"label": label, "exitcode": result.returncode, "selected": len(listing.read_text().splitlines()),
            "destination": str(destination), "reports": {key: str(directory / f"{key}.txt") for key in counts},
            "counts": counts}


def main() -> None:
    status("waiting_for_downloads")
    while active("gameplay_v8_copy_resume_20261006") or active("gameplay_v8_raw_recovery_20261006"):
        time.sleep(30)
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir()
    selected = set((SYNC / "selected_files.txt").read_text().splitlines())
    recovery = set((SYNC / "raw_recovery_files_r1.txt").read_text().splitlines())
    if len(selected) != 1081 or len(recovery) != 8 or not recovery <= selected:
        raise RuntimeError("Selected or recovered population changed")
    main_list = OUT / "main_files.txt"
    main_list.write_text("".join(name + "\n" for name in sorted(selected - recovery)))
    recovered_list = OUT / "recovered_files.txt"
    recovered_list.write_text("".join(name + "\n" for name in sorted(recovery)))
    status("checking_main", selected=len(selected - recovery))
    main = check("main", MAIN, main_list)
    status("checking_recovered", selected=len(recovery), main_exitcode=main["exitcode"])
    recovered = check("recovered", RECOVERED, recovered_list)
    result = {"status": "pass" if main["exitcode"] == recovered["exitcode"] == 0 else "fail",
              "selected_total": len(selected), "main": main, "recovered": recovered}
    write(OUT / "result.json", result)
    status("complete" if result["status"] == "pass" else "failed",
           result=str(OUT / "result.json"))
    print(json.dumps(result), flush=True)
    if result["status"] != "pass":
        raise RuntimeError("Composite raw checksum failed; inspect component reports")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
