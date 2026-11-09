#!/usr/bin/env python3
"""Lay out full10 scores from score_gameplay_pipeline_inputs.py for the gate scripts.

evaluate_gameplay_pipeline_fp_impact.py reads ``<view>/<match>/scores.csv`` and
``score_metadata.json`` (duration and window length). This writes that view with
symlinks to the scores, as the 2026-10-07 production validation did
(workflows/qwen_v8_production_20261007/validate_single_stage_when_ready.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_view(scores: Path, view: Path, expected_matches: int = 10) -> list[str]:
    metas = sorted(scores.glob("*/metadata.json"))
    if len(metas) != expected_matches:
        raise ValueError(f"Expected {expected_matches} scored matches under {scores}; found {len(metas)}")
    checkpoints = {json.loads(path.read_text())["checkpoint_sha256"] for path in metas}
    if len(checkpoints) != 1:
        raise ValueError(f"Scores come from more than one checkpoint: {sorted(checkpoints)}")
    view.mkdir(parents=True, exist_ok=False)
    matches = []
    for path in metas:
        meta = json.loads(path.read_text())
        match = Path(meta["source_manifest"]).parent.name
        source = (path.parent / "scores.csv").resolve()
        folder = view / match
        folder.mkdir()
        (folder / "scores.csv").symlink_to(source)
        info = meta["video_info"]
        (folder / "score_metadata.json").write_text(json.dumps({
            "timing": {"duration_sec": info["frame_count"] / info["fps"]},
            "window_sec": 4, "exact_input_metadata": str(path.resolve()),
            "checkpoint_sha256": meta["checkpoint_sha256"], "scores_sha256": sha256(source),
        }, indent=2) + "\n")
        matches.append(match)
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="score_gameplay_pipeline_inputs.py --output-root")
    parser.add_argument("--output", type=Path, required=True, help="new directory for the view")
    args = parser.parse_args()
    print(json.dumps({"view": str(args.output), "matches": make_view(args.scores, args.output)}, indent=2))


if __name__ == "__main__":
    main()
