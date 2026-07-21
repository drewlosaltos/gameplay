"""Tests for exact gameplay legacy-image reuse validation."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_gameplay_reuse.py"
SPEC = importlib.util.spec_from_file_location("validate_gameplay_reuse", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_fixture(tmp_path: Path, hardlink: bool = True) -> tuple[Path, Path]:
    reuse_root = tmp_path / "v5"
    dataset_root = tmp_path / "v6"
    for root in (reuse_root, dataset_root):
        (root / "metadata").mkdir(parents=True)
        (root / "metadata" / "summary.json").write_text(
            json.dumps({"long_edge": 1024, "jpeg_quality": 88}), encoding="utf-8"
        )
    source = reuse_root / "train" / "gameplay" / "source.jpg"
    target = dataset_root / "train" / "timeout" / "target.jpg"
    source.parent.mkdir(parents=True)
    target.parent.mkdir(parents=True)
    source.write_bytes(b"pixels")
    if hardlink:
        os.link(source, target)
    else:
        target.write_bytes(b"pixels")
    source_row = {"video_id": "match", "time_sec": 5.0, "path": "train/gameplay/source.jpg"}
    target_row = {"video_id": "match", "time_sec": 5.0, "path": "train/timeout/target.jpg"}
    (reuse_root / "metadata" / "images.jsonl").write_text(
        json.dumps(source_row) + "\n", encoding="utf-8"
    )
    (dataset_root / "metadata" / "images.jsonl").write_text(
        json.dumps(target_row) + "\n", encoding="utf-8"
    )
    (dataset_root / "metadata" / "summary.json").write_text(
        json.dumps(
            {
                "long_edge": 1024,
                "jpeg_quality": 88,
                "reuse_summary": {
                    "source_dataset": str(reuse_root.resolve()),
                    "reused_image_count": 1,
                    "hardlink_count": 1,
                    "copied_count": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    return dataset_root, reuse_root


def test_validate_reuse_proves_inode_identity(tmp_path: Path) -> None:
    dataset_root, reuse_root = write_fixture(tmp_path, hardlink=True)

    report = MODULE.validate_reuse(dataset_root, reuse_root)

    assert report["status"] == "passed"
    assert report["verified_hardlink_count"] == 1
    assert report["reused_video_count"] == 1


def test_validate_reuse_rejects_byte_copy(tmp_path: Path) -> None:
    dataset_root, reuse_root = write_fixture(tmp_path, hardlink=False)

    with pytest.raises(ValueError, match="Reuse counts do not match summary"):
        MODULE.validate_reuse(dataset_root, reuse_root)
