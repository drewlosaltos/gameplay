#!/usr/bin/env python3
"""Preserve r1 and build a clean v8 dataset after detecting early raw reads."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006")
OLD_WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1")
NEW_WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r2")
OLD_DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006")
NEW_DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")
RECOVERED = Path("/mnt/t/data/vball/skyball/raw_video/indoor_endline/v8_recovered_20261006_r1")
RECOVERY_LIST = Path("/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/raw_recovery_files_r1.txt")
STATUS = ROOT / "r2_clone_status.json"


def status(stage: str, **details: object) -> None:
    payload = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **details}
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(STATUS)


def updated_rows(source: Path, target: Path, overrides: dict[str, str]) -> int:
    count = 0
    with source.open() as incoming, target.open("x") as outgoing:
        for line in incoming:
            row = json.loads(line)
            override = overrides.get(row["video_id"])
            if override:
                row["raw_video_path"] = override
            outgoing.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


def main() -> None:
    if NEW_WORK.exists() or NEW_DATA.exists():
        raise FileExistsError("R2 work or dataset root already exists")
    audit = json.loads((ROOT / "early_raw_read_audit.json").read_text())
    suspect = set(audit["suspect_video_ids"])
    recover_names = RECOVERY_LIST.read_text().splitlines()
    if len(recover_names) != 8 or len(suspect) != 11:
        raise RuntimeError("Early-read audit or raw-recovery list changed")
    overrides = {Path(name).stem: str(RECOVERED / name) for name in recover_names}
    old_plan = json.loads((OLD_WORK / "plan.json").read_text())
    NEW_WORK.mkdir(parents=True)
    NEW_DATA.mkdir(parents=True)
    status("linking_verified_images", linked=0, suspect_sources=len(suspect))
    linked = 0
    excluded = 0
    for split in ("train", "val"):
        for source_dir, _, names in os.walk(OLD_DATA / split):
            relative_dir = Path(source_dir).relative_to(OLD_DATA)
            target_dir = NEW_DATA / relative_dir
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                if not name.endswith(".jpg"):
                    raise RuntimeError(f"Unexpected dataset file: {Path(source_dir) / name}")
                video_id = name.rsplit("_w", 1)[0]
                if video_id in suspect:
                    excluded += 1
                    continue
                os.link(Path(source_dir) / name, target_dir / name)
                linked += 1
                if linked % 10000 == 0:
                    status("linking_verified_images", linked=linked, excluded=excluded,
                           suspect_sources=len(suspect))
    plan = dict(old_plan)
    plan.update(version="v8_val50_v2", dataset=str(NEW_DATA), work=str(NEW_WORK),
                reused_early_raw_reprobe=str(ROOT / "reused_early_raw_reprobe_r1.json"),
                suspect_image_audit=str(ROOT / "suspect_image_audit.json"))
    (NEW_WORK / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (NEW_WORK / "raw_overrides.json").write_text(json.dumps(overrides, indent=2) + "\n")
    status("rewriting_manifests", linked=linked, excluded=excluded)
    rows = updated_rows(OLD_WORK / "planned_windows.jsonl",
                        NEW_WORK / "planned_windows.jsonl", overrides)
    if rows != plan["planned_new_windows"]:
        raise RuntimeError(f"Planned-window count changed: {rows}")
    windows = NEW_WORK / "windows_by_video"
    windows.mkdir()
    for source in sorted((OLD_WORK / "windows_by_video").glob("*.jsonl")):
        target = windows / source.name
        if source.stem in overrides:
            updated_rows(source, target, overrides)
        else:
            target.symlink_to(source)
    checks = NEW_WORK / "video_checks"
    checks.mkdir()
    copied = 0
    for source in (OLD_WORK / "video_checks").glob("*.json"):
        if source.stem in suspect or source.stem in overrides:
            continue
        shutil.copy2(source, checks / source.name)
        copied += 1
    (NEW_WORK / "base_reuse_complete.json").write_text(json.dumps({
        "status": "complete", "source_dataset": str(OLD_DATA), "dataset": str(NEW_DATA),
        "verified_hardlinks_from_r1": linked, "excluded_existing_images": excluded,
        "suspect_sources_requiring_fresh_extraction": sorted(suspect),
        "raw_source_overrides": overrides,
        "original_reuse_report": str(OLD_WORK / "base_reuse_complete.json"),
        "audit": str(ROOT / "early_raw_read_audit.json"),
    }, indent=2) + "\n")
    result = {"status": "complete", "dataset": str(NEW_DATA), "work": str(NEW_WORK),
              "linked_verified_images": linked, "excluded_existing_images": excluded,
              "copied_video_checks": copied, "planned_windows": rows,
              "suspect_sources": sorted(suspect), "raw_overrides": overrides}
    (ROOT / "r2_clone_complete.json").write_text(json.dumps(result, indent=2) + "\n")
    status("complete", result=str(ROOT / "r2_clone_complete.json"))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        status("failed", error=str(exc))
        raise
