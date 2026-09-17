"""Tests for extracted gameplay dataset validation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validate_gameplay_dataset.py"
SPEC = importlib.util.spec_from_file_location("validate_gameplay_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write compact fixture records."""
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_window(video: str, population: str, cohort: str | None) -> dict:
    """Create one valid four-frame fixture window."""
    stem = Path(video).stem
    return {
        "window_id": f"{stem}_w000000_t0000000000",
        "source_video": video,
        "video_id": stem,
        "split": "train" if population == "train" else "val",
        "split_population": population,
        "validation_cohort": cohort,
        "class_name": "gameplay",
        "binary_label": 1,
        "start_sec": 0.0,
        "times_sec": [0.0, 1.0, 2.0, 3.0],
        "frame_paths": [f"{'train' if population == 'train' else 'val'}/gameplay/{stem}_{i}.jpg" for i in range(4)],
    }


def test_validate_dataset_confirms_staged_split_and_frozen_v2_windows(tmp_path: Path) -> None:
    """A complete staged fixture should pass every dataset invariant."""
    dataset = tmp_path / "v5"
    metadata = dataset / "metadata"
    metadata.mkdir(parents=True)
    windows = [
        make_window("train.mp4", "train", None),
        make_window("legacy.mp4", "val", "v2_fixed"),
        make_window("expanded.mp4", "val", "v5_expanded"),
    ]
    images = []
    for window in windows:
        for frame_index, frame_path in enumerate(window["frame_paths"]):
            path = dataset / frame_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            images.append({"window_id": window["window_id"], "path": frame_path, "frame_index": frame_index})
    write_jsonl(metadata / "windows.jsonl", windows)
    write_jsonl(metadata / "images.jsonl", images)
    (metadata / "class_to_idx.json").write_text('{"gameplay": 0}\n', encoding="utf-8")
    (metadata / "summary.json").write_text(
        json.dumps(
            {
                "window_count": 3,
                "image_count": 12,
                "video_count": 3,
                "sampling_policy": {
                    "train": {"gameplay_stride_sec": 20.0, "negative_stride_sec": 10.0},
                    "v2_fixed": {"gameplay_stride_sec": 20.0, "negative_stride_sec": 1.0},
                    "v5_expanded": {"gameplay_stride_sec": 10.0, "negative_stride_sec": 10.0},
                },
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base"
    base.mkdir()
    write_jsonl(base / "windows.jsonl", [{**windows[1], "validation_cohort": None}])
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "sampling_policy": {
                    "train": {"gameplay_stride_sec": 20.0, "negative_stride_sec": 10.0},
                    "v2_fixed": {"gameplay_stride_sec": 20.0, "negative_stride_sec": 1.0},
                    "v5_expanded": {"gameplay_stride_sec": 10.0, "negative_stride_sec": 10.0},
                },
                "v2_fixed_videos": ["legacy.mp4"],
                "videos": [{"source_video": "expanded.mp4"}],
            }
        ),
        encoding="utf-8",
    )

    report = MODULE.validate_dataset(dataset, base, manifest)

    assert report["status"] == "passed"
    assert report["population_video_counts"] == {"train": 1, "v2_fixed": 1, "v5_expanded": 1}


def test_validate_dataset_rejects_split_leakage(tmp_path: Path) -> None:
    """The same source video cannot occur in train and validation."""
    train = make_window("duplicate.mp4", "train", None)
    val = make_window("duplicate.mp4", "val", "v5_expanded")

    try:
        MODULE.validate_population_membership([train, val], {"legacy"}, {"duplicate"})
    except ValueError as exc:
        assert "multiple populations" in str(exc)
    else:
        raise AssertionError("Expected split leakage to fail")


def test_validate_source_held_out_dataset_checks_beach_split_contract(tmp_path: Path) -> None:
    """A beach-only source-held-out dataset should pass the generic audit."""
    dataset = tmp_path / "v7"
    metadata = dataset / "metadata"
    metadata.mkdir(parents=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    windows = []
    images = []
    for video, split, cohort in [
        ("beach_train.mp4", "train", None),
        ("beach_val.mp4", "val", "beach_v7"),
    ]:
        raw_path = raw / video
        raw_path.touch()
        stem = Path(video).stem
        frame_paths = [f"{split}/gameplay/{stem}_s{index}.jpg" for index in range(4)]
        window = {
            "window_id": f"{stem}_w000000_t0000000000",
            "source_video": video,
            "video_id": stem,
            "split": split,
            "validation_cohort": cohort,
            "class_name": "gameplay",
            "binary_label": 1,
            "start_sec": 0.0,
            "end_sec": 4.0,
            "times_sec": [0.0, 1.0, 2.0, 3.0],
            "frame_paths": frame_paths,
            "stable_start_sec": 0.0,
            "stable_end_sec": 4.0,
            "start_exclude_sec": 0.0,
            "end_exclude_sec": 0.0,
            "boundary_margin_sec": 0.0,
            "sampling_seed": 7,
            "raw_video_path": str(raw_path),
        }
        windows.append(window)
        for frame_index, frame_path in enumerate(frame_paths):
            path = dataset / frame_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            images.append(
                {
                    "image_id": f"{stem}_s{frame_index}",
                    "window_id": window["window_id"],
                    "path": frame_path,
                }
            )
    write_jsonl(metadata / "windows.jsonl", windows)
    write_jsonl(metadata / "images.jsonl", images)
    (metadata / "class_to_idx.json").write_text('{"gameplay": 0}\n', encoding="utf-8")
    (metadata / "summary.json").write_text(
        json.dumps({"window_count": 2, "image_count": 8, "video_count": 2, "seed": 7}),
        encoding="utf-8",
    )
    population = tmp_path / "population.json"
    population.write_text(
        json.dumps({"videos": ["beach_train", "beach_val"]}),
        encoding="utf-8",
    )
    val = tmp_path / "val.json"
    val.write_text(json.dumps({"videos": ["beach_val"]}), encoding="utf-8")

    report = MODULE.validate_source_held_out_dataset(dataset, population, val)

    assert report["status"] == "passed"
    assert report["train_video_count"] == 1
    assert report["val_video_count"] == 1
    assert report["source_split_leakage_count"] == 0
