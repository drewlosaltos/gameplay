"""Tests for versioned gameplay dataset sampling policies."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_gameplay_v0_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_gameplay_dataset", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


LabelRegion = MODULE.LabelRegion


def region(video_id: str, class_dir: str, end_sec: float = 34.0) -> LabelRegion:
    """Create one stable synthetic label region."""
    return LabelRegion(
        row_index=0,
        source_video=f"{video_id}.mp4",
        video_id=video_id,
        category=class_dir,
        class_dir=class_dir,
        start_sec=0.0,
        end_sec=end_sec,
        start_exclude_sec=0.0,
        end_exclude_sec=0.0,
    )


def test_staged_sampling_preserves_legacy_and_uniformly_samples_expanded_validation() -> None:
    """Each v5 population should use its declared temporal sampling policy."""
    windows = MODULE.build_staged_windows(
        regions=[
            region("train_negative", "timeout"),
            region("train_gameplay", "gameplay"),
            region("legacy_negative", "timeout"),
            region("expanded_negative", "timeout"),
            region("expanded_gameplay", "gameplay"),
        ],
        legacy_val_videos={"legacy_negative"},
        expanded_val_videos={"expanded_negative", "expanded_gameplay"},
        clip_seconds=4,
        train_gameplay_stride=20.0,
        train_negative_stride=10.0,
        legacy_gameplay_stride=20.0,
        legacy_negative_stride=1.0,
        expanded_val_stride=10.0,
    )

    starts = {}
    for window in windows:
        starts.setdefault(window["video_id"], []).append(window["start_sec"])

    assert starts["train_negative"] == [0.0, 10.0, 20.0, 30.0]
    assert starts["train_gameplay"] == [0.0, 20.0]
    assert starts["legacy_negative"] == [float(value) for value in range(31)]
    assert starts["expanded_negative"] == [0.0, 10.0, 20.0, 30.0]
    assert starts["expanded_gameplay"] == [0.0, 10.0, 20.0, 30.0]


def test_staged_sampling_marks_validation_cohorts_without_split_leakage() -> None:
    """Legacy and expanded validation videos must remain distinct from training."""
    windows = MODULE.build_staged_windows(
        regions=[
            region("train", "gameplay"),
            region("legacy", "gameplay"),
            region("expanded", "gameplay"),
        ],
        legacy_val_videos={"legacy"},
        expanded_val_videos={"expanded"},
        clip_seconds=4,
        train_gameplay_stride=20.0,
        train_negative_stride=10.0,
        legacy_gameplay_stride=20.0,
        legacy_negative_stride=1.0,
        expanded_val_stride=10.0,
        legacy_val_cohort="beach_v7",
    )

    populations = {
        window["video_id"]: (window["split_population"], window["validation_cohort"])
        for window in windows
    }
    assert populations == {
        "train": ("train", None),
        "legacy": ("val", "beach_v7"),
        "expanded": ("val", "v5_expanded"),
    }


def test_staged_sampling_rejects_overlapping_validation_cohorts() -> None:
    """A source video cannot belong to both validation cohorts."""
    try:
        MODULE.build_staged_windows(
            regions=[region("overlap", "gameplay")],
            legacy_val_videos={"overlap"},
            expanded_val_videos={"overlap"},
            clip_seconds=4,
            train_gameplay_stride=20.0,
            train_negative_stride=10.0,
            legacy_gameplay_stride=20.0,
            legacy_negative_stride=1.0,
            expanded_val_stride=10.0,
        )
    except ValueError as exc:
        assert "both validation cohorts" in str(exc)
    else:
        raise AssertionError("Expected overlapping validation cohorts to fail")


def test_extract_video_set_collects_parallel_results() -> None:
    """Parallel extraction should retain deterministic per-video results."""
    raw_videos = {"a": Path("a.mp4"), "b": Path("b.mp4")}
    timestamp_outputs = {
        "a": {0.0: [Path("a0.jpg")], 1.0: [Path("a1.jpg")]},
        "b": {0.0: [Path("b0.jpg")]},
    }

    def fake_extract(
        video_path: Path,
        outputs: dict[float, list[Path]],
        long_edge: int,
        jpeg_quality: int,
    ) -> tuple[int, list[float]]:
        assert long_edge == 1024
        assert jpeg_quality == 88
        return len(outputs), [99.0] if video_path.stem == "b" else []

    extracted, failures = MODULE.extract_video_set(
        video_ids=["a", "b"],
        raw_videos=raw_videos,
        timestamp_outputs=timestamp_outputs,
        long_edge=1024,
        jpeg_quality=88,
        workers=2,
        extractor=fake_extract,
    )

    assert extracted == {"a": 2, "b": 1}
    assert failures == {"b": [99.0]}


def test_resolve_cached_frame_clamps_near_end_boundary(tmp_path: Path) -> None:
    """A near-boundary request should reuse the final real decoded frame."""
    for frame_index in range(3):
        (tmp_path / f"{frame_index:08d}.jpg").write_bytes(str(frame_index).encode())

    resolved = MODULE.resolve_cached_frame(tmp_path, frame_index=4)

    assert resolved == tmp_path / "00000002.jpg"


def test_resolve_cached_frame_rejects_distant_or_internal_gap(tmp_path: Path) -> None:
    """Boundary clamping must not conceal a distant or internal decode gap."""
    for frame_index in [0, 1, 3, 4]:
        (tmp_path / f"{frame_index:08d}.jpg").write_bytes(str(frame_index).encode())

    assert MODULE.resolve_cached_frame(tmp_path, frame_index=2) is None
    assert MODULE.resolve_cached_frame(tmp_path, frame_index=7) is None


def test_extract_video_frames_resume_preserves_existing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resume extraction should fill only missing outputs and preserve existing JPEGs."""
    dataset_root = tmp_path / "dataset"
    existing = dataset_root / "train" / "gameplay" / "existing.jpg"
    missing = dataset_root / "train" / "gameplay" / "missing.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"preserve-me")

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        cache_dir = Path(command[-1]).parent
        for frame_index in range(3):
            (cache_dir / f"{frame_index:08d}.jpg").write_bytes(f"frame-{frame_index}".encode())

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)

    extracted, failures = MODULE.extract_video_frames(
        video_path=tmp_path / "source.mp4",
        timestamp_to_outputs={0.0: [existing], 2.0: [missing]},
        long_edge=1024,
        jpeg_quality=88,
    )

    assert extracted == 1
    assert failures == []
    assert existing.read_bytes() == b"preserve-me"
    assert missing.read_bytes() == b"frame-2"


def test_reuse_legacy_images_matches_video_timestamp_across_new_paths(tmp_path: Path) -> None:
    """A relabeled/reindexed window may reuse pixels from the same raw timestamp."""
    reuse_root = tmp_path / "v5"
    output_root = tmp_path / "v6"
    source_jpeg = reuse_root / "train" / "gameplay" / "old_window_s0.jpg"
    source_jpeg.parent.mkdir(parents=True)
    source_jpeg.write_bytes(b"legacy-frame")
    raw_video = tmp_path / "raw" / "match.mp4"
    raw_video.parent.mkdir()
    raw_video.write_bytes(b"video-placeholder")
    metadata = reuse_root / "metadata"
    metadata.mkdir()
    (metadata / "summary.json").write_text(
        json.dumps({"long_edge": 1024, "jpeg_quality": 88}),
        encoding="utf-8",
    )
    (metadata / "images.jsonl").write_text(
        json.dumps(
            {
                "video_id": "match",
                "time_sec": 42.0,
                "raw_video_path": str(raw_video),
                "path": "train/gameplay/old_window_s0.jpg",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    target_a = output_root / "train" / "timeout" / "new_window_s0.jpg"
    target_b = output_root / "train" / "timeout" / "new_window_s1.jpg"

    summary = MODULE.reuse_legacy_images(
        reuse_root=reuse_root,
        output_root=output_root,
        raw_videos={"match": raw_video},
        timestamp_outputs={"match": {42.0: [target_a, target_b], 43.0: [output_root / "missing.jpg"]}},
        long_edge=1024,
        jpeg_quality=88,
    )

    assert target_a.read_bytes() == b"legacy-frame"
    assert target_b.read_bytes() == b"legacy-frame"
    assert os.stat(source_jpeg).st_ino == os.stat(target_a).st_ino
    assert summary["reused_image_count"] == 2
    assert summary["reused_unique_video_timestamp_count"] == 1
    assert summary["hardlink_count"] == 2
    assert summary["copied_count"] == 0


def test_reuse_legacy_images_rejects_incompatible_resolution(tmp_path: Path) -> None:
    """Legacy JPEGs must not be reused when extraction settings differ."""
    reuse_root = tmp_path / "v5"
    metadata = reuse_root / "metadata"
    metadata.mkdir(parents=True)
    (metadata / "summary.json").write_text(
        json.dumps({"long_edge": 512, "jpeg_quality": 88}),
        encoding="utf-8",
    )
    (metadata / "images.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="extraction settings differ"):
        MODULE.reuse_legacy_images(
            reuse_root=reuse_root,
            output_root=tmp_path / "v6",
            raw_videos={},
            timestamp_outputs={},
            long_edge=1024,
            jpeg_quality=88,
        )


def test_beach_labels_are_preserved_as_binary_non_gameplay_classes() -> None:
    """Beach-only stoppage labels should map into an existing class folder."""
    for label in [
        "side switch",
        "court maintenance break",
        "referee mark inspection",
    ]:
        assert MODULE.class_dir_for(label) == "timeout"
