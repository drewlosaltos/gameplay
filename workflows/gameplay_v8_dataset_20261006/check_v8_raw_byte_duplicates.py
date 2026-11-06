#!/usr/bin/env python3
"""Audit exact-byte raw-video duplication across the published v8 cohorts."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

DATA = Path("/mnt/t/data/vball/skyball/gameplay/v8_val50_20261006_r2")
OUT = Path("/mnt/t/output/skyball_gameplay/v8_workflow_20261006/raw_byte_duplicate_preflight.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    summary_path = DATA / "metadata/summary.json"
    windows_path = DATA / "metadata/windows.jsonl"
    summary = json.loads(summary_path.read_text())
    cohort = {name: label for label, names in summary["source_videos_by_cohort"].items()
              for name in names}
    if len(cohort) != summary["video_count"]:
        raise RuntimeError("Cohort membership repeats a video ID")
    paths = {}
    with windows_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            vid, path = row["video_id"], row["raw_video_path"]
            if vid in paths and paths[vid] != path:
                raise RuntimeError(f"One video ID maps to multiple raw files: {vid}")
            paths[vid] = path
    if set(paths) != set(cohort):
        raise RuntimeError("Manifest videos differ from published cohort IDs")
    by_path = defaultdict(list)
    by_size = defaultdict(list)
    for vid, raw_path in paths.items():
        path = Path(raw_path)
        by_path[str(path.resolve())].append((vid, cohort[vid]))
        by_size[path.stat().st_size].append((vid, cohort[vid]))
    shared_paths = {path: members for path, members in by_path.items() if len(members) > 1}
    repeated_sizes = {size: members for size, members in by_size.items() if len(members) > 1}
    cross_cohort_sizes = {size: members for size, members in repeated_sizes.items()
                          if len({member[1] for member in members}) > 1}
    if shared_paths or cross_cohort_sizes:
        raise RuntimeError(f"Potential raw-video leakage: {len(shared_paths)} shared paths, "
                           f"{len(cross_cohort_sizes)} cross-cohort equal-size groups")
    report = {
        "status": "pass", "scope": "Exact-byte duplicate exclusion by unique raw file size; "
        "re-encodes of the same visual content are outside this check",
        "dataset": str(DATA), "windows_sha256": sha256(windows_path),
        "summary_sha256": sha256(summary_path),
        "videos": len(paths),
        "cohorts": {label: sum(value == label for value in cohort.values())
                    for label in summary["source_videos_by_cohort"]},
        "shared_raw_paths": len(shared_paths),
        "equal_size_groups": len(repeated_sizes),
        "cross_cohort_equal_size_groups": len(cross_cohort_sizes),
    }
    temporary = OUT.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(OUT)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
