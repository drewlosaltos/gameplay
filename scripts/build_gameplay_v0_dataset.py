#!/usr/bin/env python3
"""Sync raw SkyBall gameplay videos and build an ImageNet-style frame dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

try:
    import ffmpegcv
except ImportError:  # pragma: no cover - environment dependent
    ffmpegcv = None


DEFAULT_WORKBOOK = Path("/mnt/t/data/vball/skyball/gameplay/ground_truth/v1.xlsx")
DEFAULT_RAW_ROOT = Path("/mnt/t/data/vball/skyball/raw_video")
DEFAULT_OUTPUT_ROOT = Path("/mnt/t/data/vball/skyball/gameplay/v0")
DEFAULT_REMOTE = "googdrive:skyball/raw_video"

CLASS_DIRS = {
    "gameplay": "gameplay",
    "timeout": "timeout",
    "side switch": "timeout",
    "court maintenance break": "timeout",
    "referee mark inspection": "timeout",
    "set break": "setbreak",
    "set_break": "setbreak",
    "pre match": "prematch",
    "pre-match": "prematch",
    "pre_match": "prematch",
    "post match": "postmatch",
    "post-match": "postmatch",
    "post_match": "postmatch",
}


@dataclass(frozen=True)
class LabelRegion:
    row_index: int
    source_video: str
    video_id: str
    category: str
    class_dir: str
    start_sec: float
    end_sec: float
    start_exclude_sec: float
    end_exclude_sec: float

    @property
    def stable_start_sec(self) -> float:
        return self.start_sec + self.start_exclude_sec

    @property
    def stable_end_sec(self) -> float:
        return self.end_sec - self.end_exclude_sec

    @property
    def stable_duration_sec(self) -> float:
        return max(0.0, self.stable_end_sec - self.stable_start_sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync-raw", help="rclone workbook raw videos into the local raw tree")
    sync.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    sync.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    sync.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sync.add_argument("--remote", default=DEFAULT_REMOTE)
    sync.add_argument("--remote-list-cache", type=Path)
    sync.add_argument(
        "--video-manifest",
        type=Path,
        help="optional JSON manifest limiting the workbook to its top-level videos list",
    )
    sync.add_argument("--dry-run", action="store_true")
    sync.add_argument("--skip-duration-check", action="store_true")

    validate = subparsers.add_parser("validate-raw", help="validate local raw-video durations")
    validate.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    validate.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    validate.add_argument("--metadata-dir", type=Path, required=True)
    validate.add_argument("--tolerance-sec", type=float, default=0.25)
    validate.add_argument(
        "--video-manifest",
        type=Path,
        help="optional JSON manifest limiting the workbook to its top-level videos list",
    )

    extract = subparsers.add_parser("extract", help="extract sampled 4-frame windows into ImageNet folders")
    extract.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    extract.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    extract.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    output_mode = extract.add_mutually_exclusive_group()
    output_mode.add_argument("--force", action="store_true")
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="preserve existing images, extract only missing outputs, and rewrite metadata",
    )
    extract.add_argument("--seed", type=int, default=20260628)
    extract.add_argument("--val-fraction", type=float, default=0.20)
    validation_source = extract.add_mutually_exclusive_group()
    validation_source.add_argument(
        "--static-val-metadata",
        type=Path,
        help="metadata dir or windows.jsonl from a base dataset whose validation source videos should be preserved",
    )
    validation_source.add_argument(
        "--val-video-manifest",
        type=Path,
        help="JSON manifest containing the exact held-out validation videos",
    )
    extract.add_argument(
        "--video-manifest",
        type=Path,
        help="optional JSON manifest limiting the workbook to its top-level videos list",
    )
    extract.add_argument(
        "--val-cohort-name",
        default="v2_fixed",
        help="validation cohort label written for the base or explicitly held-out videos",
    )
    extract.add_argument(
        "--train-split-name",
        default="train",
        choices=["train", "trn"],
        help="metadata split name for training records; paths always use train/ for ImageFolder compatibility",
    )
    extract.add_argument("--clip-seconds", type=int, default=4)
    extract.add_argument("--gameplay-stride-sec", type=float, default=20.0)
    extract.add_argument("--negative-stride-sec", type=float, default=1.0)
    extract.add_argument(
        "--train-negative-stride-sec",
        type=float,
        help="optional training-only negative stride; validation keeps --negative-stride-sec",
    )
    extract.add_argument(
        "--expanded-val-manifest",
        type=Path,
        help="JSON manifest containing an additional held-out validation video list",
    )
    extract.add_argument(
        "--expanded-val-stride-sec",
        type=float,
        default=10.0,
        help="uniform stride for gameplay and non-gameplay in the expanded validation cohort",
    )
    extract.add_argument("--prematch-gameplay-zone-sec", type=float, default=20.0)
    extract.add_argument("--other-transition-zone-sec", type=float, default=5.0)
    extract.add_argument("--long-edge", type=int, default=1024)
    extract.add_argument("--jpeg-quality", type=int, default=88)
    extract.add_argument(
        "--reuse-images-from",
        type=Path,
        help=(
            "existing gameplay dataset root whose JPEGs may be hardlinked by "
            "source-video and timestamp when extraction settings match"
        ),
    )
    extract.add_argument(
        "--extract-workers",
        type=int,
        default=4,
        help="number of source videos to decode concurrently; this does not change sampling",
    )
    return parser.parse_args()


def normalize_video_id(value: object) -> str:
    return Path(str(value).strip()).stem


def normalize_category(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def class_dir_for(category: object) -> str:
    normalized = normalize_category(category)
    if normalized not in CLASS_DIRS:
        raise ValueError(f"Unsupported gameplay class: {category!r}")
    return CLASS_DIRS[normalized]


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


def load_video_manifest(path: Path) -> set[str]:
    """Load normalized source-video IDs from a top-level ``videos`` list."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = payload.get("videos", [])
    video_ids = {
        normalize_video_id(entry.get("source_video", "") if isinstance(entry, dict) else entry)
        for entry in videos
    }
    video_ids.discard("")
    if not video_ids:
        raise ValueError(f"No videos found in {path}")
    return video_ids


def read_labels(workbook: Path, video_ids: set[str] | None = None) -> pd.DataFrame:
    labels = pd.read_excel(workbook)
    labels = labels.dropna(subset=["video", "category", "start_sec", "end_sec"]).copy()
    labels["source_video"] = labels["video"].astype(str).str.strip()
    labels["video_id"] = labels["source_video"].map(normalize_video_id)
    labels["category"] = labels["category"].astype(str).str.strip()
    labels["category_norm"] = labels["category"].map(normalize_category)
    labels["class_dir"] = labels["category"].map(class_dir_for)
    labels["start_sec"] = labels["start_sec"].astype(float)
    labels["end_sec"] = labels["end_sec"].astype(float)
    if video_ids is not None:
        available = set(labels["video_id"])
        missing = sorted(video_ids - available)
        if missing:
            raise ValueError(
                f"{len(missing)} manifest videos are missing from {workbook}: "
                + ", ".join(missing)
            )
        labels = labels[labels["video_id"].isin(video_ids)].copy()
    labels = labels.sort_values(["video_id", "start_sec", "end_sec"]).reset_index(drop=False)
    labels = labels.rename(columns={"index": "row_index"})
    return labels


def workbook_video_names(workbook: Path, video_ids: set[str] | None = None) -> set[str]:
    labels = read_labels(workbook, video_ids)
    return set(labels["source_video"])


def local_raw_videos(raw_root: Path) -> dict[str, Path]:
    videos: dict[str, Path] = {}
    for path in raw_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp4", ".mov", ".mkv"}:
            videos.setdefault(path.stem, path)
    return videos


def list_remote_files(remote: str, cache_path: Path) -> list[str]:
    if cache_path.exists():
        return [line.strip() for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rclone",
        "lsf",
        remote,
        "--recursive",
        "--files-only",
        "--include",
        "*.mp4",
        "--include",
        "*.MP4",
        "--include",
        "*.mov",
        "--include",
        "*.MOV",
        "--include",
        "*.mkv",
        "--include",
        "*.MKV",
    ]
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE)
    cache_path.write_text(result.stdout, encoding="utf-8")
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def choose_remote_paths(remote_paths: list[str], wanted_names: set[str]) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for rel_path in remote_paths:
        name = Path(rel_path).name
        if name in wanted_names:
            by_name[name].append(rel_path)

    chosen: dict[str, str] = {}
    duplicates: dict[str, list[str]] = {}
    for name in sorted(wanted_names):
        candidates = sorted(by_name.get(name, []))
        if not candidates:
            continue
        if len(candidates) > 1:
            duplicates[name] = candidates
        preferred = [path for path in candidates if path.startswith("indoor_endline/youth/")]
        chosen[name] = preferred[0] if preferred else candidates[0]
    return chosen, duplicates


def run_sync_raw(args: argparse.Namespace) -> None:
    metadata_dir = args.output_root.parent / f"{args.output_root.name}_raw_sync"
    cache_path = args.remote_list_cache or (metadata_dir / "remote_raw_video_files.txt")
    video_ids = load_video_manifest(args.video_manifest) if args.video_manifest else None
    wanted = workbook_video_names(args.workbook, video_ids)
    remote_paths = list_remote_files(args.remote, cache_path)
    chosen, duplicates = choose_remote_paths(remote_paths, wanted)
    missing = sorted(wanted - set(chosen))

    metadata_dir.mkdir(parents=True, exist_ok=True)
    files_from = metadata_dir / "rclone_raw_videos_files_from.txt"
    files_from.write_text("".join(f"{path}\n" for path in chosen.values()), encoding="utf-8")
    sync_summary = {
        "workbook": str(args.workbook),
        "remote": args.remote,
        "raw_root": str(args.raw_root),
        "wanted_count": len(wanted),
        "matched_count": len(chosen),
        "missing": missing,
        "duplicates": duplicates,
        "files_from": str(files_from),
    }
    (metadata_dir / "raw_video_sync_summary.json").write_text(
        json.dumps(sync_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(sync_summary, indent=2, sort_keys=True))
    if missing:
        raise SystemExit(f"Missing {len(missing)} workbook videos on remote; not syncing incomplete set.")
    if args.dry_run:
        return

    args.raw_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        "rclone",
        "copy",
        args.remote,
        str(args.raw_root),
        "--files-from",
        str(files_from),
        "--ignore-existing",
        "--create-empty-src-dirs",
        "--transfers",
        "4",
        "--checkers",
        "8",
        "--stats",
        "30s",
        "--progress",
    ]
    subprocess.run(cmd, check=True)
    if not args.skip_duration_check:
        validate_raw_durations(args.workbook, args.raw_root, metadata_dir, video_ids=video_ids)


def ffprobe_duration_sec(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or f"ffprobe exited {result.returncode}"
        raise ValueError(message)
    return float(result.stdout.strip())


def validate_raw_durations(
    workbook: Path,
    raw_root: Path,
    metadata_dir: Path,
    tolerance_sec: float = 0.25,
    video_ids: set[str] | None = None,
) -> None:
    labels = read_labels(workbook, video_ids)
    raw_videos = local_raw_videos(raw_root)
    rows = []
    mismatches = []
    for video_id, group in labels.groupby("video_id", sort=True):
        raw_path = raw_videos.get(video_id)
        max_end_sec = float(group["end_sec"].max())
        if raw_path is None:
            row = {
                "video_id": video_id,
                "source_video": str(group["source_video"].iloc[0]),
                "raw_video_path": None,
                "workbook_max_end_sec": max_end_sec,
                "duration_sec": None,
                "margin_sec": None,
                "status": "missing",
            }
            rows.append(row)
            mismatches.append(row)
            continue
        try:
            duration_sec = ffprobe_duration_sec(raw_path)
            margin_sec = duration_sec - max_end_sec
            status = "ok" if margin_sec + tolerance_sec >= 0 else "duration_short"
            error = None
        except Exception as exc:
            duration_sec = None
            margin_sec = None
            status = "probe_failed"
            error = str(exc)
        row = {
            "video_id": video_id,
            "source_video": str(group["source_video"].iloc[0]),
            "raw_video_path": str(raw_path),
            "workbook_max_end_sec": round(max_end_sec, 3),
            "duration_sec": round(duration_sec, 3) if duration_sec is not None else None,
            "margin_sec": round(margin_sec, 3) if margin_sec is not None else None,
            "status": status,
        }
        if error:
            row["error"] = error
        rows.append(row)
        if status != "ok":
            mismatches.append(row)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    validation = {
        "workbook": str(workbook),
        "raw_root": str(raw_root),
        "tolerance_sec": tolerance_sec,
        "video_count": len(rows),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "videos": rows,
    }
    path = metadata_dir / "raw_video_duration_validation.json"
    path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: validation[k] for k in ["video_count", "mismatch_count"]}, indent=2, sort_keys=True))
    if mismatches:
        raise SystemExit(f"Raw duration validation failed for {len(mismatches)} videos; see {path}")


def transition_zone(left_category: str, right_category: str, prematch_gameplay_zone: float, other_zone: float) -> float:
    pair = {normalize_category(left_category), normalize_category(right_category)}
    if pair == {"pre match", "gameplay"}:
        return prematch_gameplay_zone
    return other_zone


def build_regions(labels: pd.DataFrame, prematch_gameplay_zone: float, other_zone: float) -> list[LabelRegion]:
    labels = labels.copy()
    labels["start_exclude_sec"] = 0.0
    labels["end_exclude_sec"] = 0.0
    for _, group in labels.groupby("video_id", sort=False):
        indices = list(group.index)
        for left, right in zip(indices, indices[1:]):
            zone = transition_zone(
                labels.loc[left, "category_norm"],
                labels.loc[right, "category_norm"],
                prematch_gameplay_zone,
                other_zone,
            )
            half = zone / 2.0
            labels.loc[left, "end_exclude_sec"] = max(float(labels.loc[left, "end_exclude_sec"]), half)
            labels.loc[right, "start_exclude_sec"] = max(float(labels.loc[right, "start_exclude_sec"]), half)

    regions: list[LabelRegion] = []
    for _, row in labels.iterrows():
        regions.append(
            LabelRegion(
                row_index=int(row["row_index"]),
                source_video=str(row["source_video"]),
                video_id=str(row["video_id"]),
                category=str(row["category"]),
                class_dir=str(row["class_dir"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                start_exclude_sec=float(row["start_exclude_sec"]),
                end_exclude_sec=float(row["end_exclude_sec"]),
            )
        )
    return regions


def sample_starts(region: LabelRegion, clip_seconds: int, gameplay_stride: float, negative_stride: float) -> list[float]:
    stride = gameplay_stride if region.class_dir == "gameplay" else negative_stride
    duration = region.stable_duration_sec
    if duration < clip_seconds:
        return []
    count = int(math.floor((duration - clip_seconds) / stride)) + 1
    return [region.stable_start_sec + i * stride for i in range(count)]


def build_windows(
    regions: list[LabelRegion],
    clip_seconds: int,
    gameplay_stride: float,
    negative_stride: float,
) -> list[dict]:
    windows: list[dict] = []
    per_video_index: Counter[str] = Counter()
    for region in regions:
        for start_sec in sample_starts(region, clip_seconds, gameplay_stride, negative_stride):
            window_index = per_video_index[region.video_id]
            per_video_index[region.video_id] += 1
            window_id = f"{region.video_id}_w{window_index:06d}_t{int(round(start_sec * 1000)):010d}"
            times_sec = [round(start_sec + offset, 3) for offset in range(clip_seconds)]
            windows.append(
                {
                    "window_id": window_id,
                    "source_video": region.source_video,
                    "video_id": region.video_id,
                    "class_name": region.class_dir,
                    "original_category": region.category,
                    "binary_label": 1 if region.class_dir == "gameplay" else 0,
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(start_sec + clip_seconds, 3),
                    "start_hms": hms(start_sec),
                    "times_sec": times_sec,
                    "source_row_index": region.row_index,
                    "region_start_sec": round(region.start_sec, 3),
                    "region_end_sec": round(region.end_sec, 3),
                    "stable_start_sec": round(region.stable_start_sec, 3),
                    "stable_end_sec": round(region.stable_end_sec, 3),
                    "start_exclude_sec": round(region.start_exclude_sec, 3),
                    "end_exclude_sec": round(region.end_exclude_sec, 3),
                    "stride_sec": gameplay_stride if region.class_dir == "gameplay" else negative_stride,
                }
            )
    return windows


def build_staged_windows(
    regions: list[LabelRegion],
    legacy_val_videos: set[str],
    expanded_val_videos: set[str],
    clip_seconds: int,
    train_gameplay_stride: float,
    train_negative_stride: float,
    legacy_gameplay_stride: float,
    legacy_negative_stride: float,
    expanded_val_stride: float,
    legacy_val_cohort: str = "v2_fixed",
) -> list[dict]:
    """Build train and validation windows with population-specific strides.

    Args:
        regions: Stable labeled regions to sample.
        legacy_val_videos: Frozen base-validation video IDs.
        expanded_val_videos: Additional held-out validation video IDs.
        clip_seconds: Number of one-Hz frames in each temporal window.
        train_gameplay_stride: Gameplay start spacing for training videos.
        train_negative_stride: Non-gameplay start spacing for training videos.
        legacy_gameplay_stride: Gameplay start spacing for frozen validation.
        legacy_negative_stride: Non-gameplay start spacing for frozen validation.
        expanded_val_stride: Class-independent start spacing for expanded validation.

    Returns:
        Windows annotated with their split population and validation cohort.

    Raises:
        ValueError: If a video belongs to both validation cohorts.
    """
    overlap = legacy_val_videos & expanded_val_videos
    if overlap:
        raise ValueError(
            "Videos cannot belong to both validation cohorts: " + ", ".join(sorted(overlap))
        )

    train_regions = [
        region
        for region in regions
        if region.video_id not in legacy_val_videos and region.video_id not in expanded_val_videos
    ]
    legacy_regions = [region for region in regions if region.video_id in legacy_val_videos]
    expanded_regions = [region for region in regions if region.video_id in expanded_val_videos]

    populations = [
        (
            build_windows(train_regions, clip_seconds, train_gameplay_stride, train_negative_stride),
            "train",
            None,
        ),
        (
            build_windows(legacy_regions, clip_seconds, legacy_gameplay_stride, legacy_negative_stride),
            "val",
            legacy_val_cohort,
        ),
        (
            build_windows(expanded_regions, clip_seconds, expanded_val_stride, expanded_val_stride),
            "val",
            "v5_expanded",
        ),
    ]
    windows: list[dict] = []
    for population_windows, split_population, validation_cohort in populations:
        for window in population_windows:
            window["split_population"] = split_population
            window["validation_cohort"] = validation_cohort
            windows.append(window)
    return sorted(windows, key=lambda window: (window["video_id"], window["start_sec"], window["window_id"]))


def choose_val_videos(windows: list[dict], seed: int, val_fraction: float) -> set[str]:
    by_video_class: dict[str, Counter[str]] = defaultdict(Counter)
    total: Counter[str] = Counter()
    for window in windows:
        by_video_class[window["video_id"]][window["class_name"]] += 1
        total[window["class_name"]] += 1
    videos = sorted(by_video_class)
    target_video_count = max(1, int(round(len(videos) * val_fraction)))
    target_by_class = {cls: count * val_fraction for cls, count in total.items()}

    rng = random.Random(seed)
    remaining = videos[:]
    rng.shuffle(remaining)
    selected: set[str] = set()
    selected_counts: Counter[str] = Counter()

    while len(selected) < target_video_count and remaining:
        best_video = None
        best_score = None
        for video in remaining:
            candidate = selected_counts + by_video_class[video]
            score = 0.0
            for cls, target in target_by_class.items():
                denom = max(target, 1.0)
                score += abs(candidate[cls] - target) / denom
            if best_score is None or score < best_score or (score == best_score and video < best_video):
                best_score = score
                best_video = video
        assert best_video is not None
        selected.add(best_video)
        selected_counts.update(by_video_class[best_video])
        remaining.remove(best_video)
    return selected


def load_static_val_videos(path: Path) -> set[str]:
    windows_path = path / "windows.jsonl" if path.is_dir() else path
    if not windows_path.exists():
        raise FileNotFoundError(f"Static validation metadata does not exist: {windows_path}")
    val_videos: set[str] = set()
    with windows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("split") == "val":
                video_id = record.get("video_id") or normalize_video_id(record.get("source_video", ""))
                if video_id:
                    val_videos.add(str(video_id))
    if not val_videos:
        raise ValueError(f"No validation videos found in {windows_path}")
    return val_videos


def load_expanded_val_videos(path: Path) -> set[str]:
    """Load normalized video IDs from a versioned expanded-validation manifest.

    Args:
        path: JSON manifest containing a top-level ``videos`` list. Entries may
            be source-video strings or objects with ``source_video``.

    Returns:
        Normalized source-video IDs.

    Raises:
        ValueError: If the manifest has no usable videos.
    """
    return load_video_manifest(path)


def resize_lanczos(image_bgr, long_edge: int) -> Image.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    width, height = image.size
    scale = long_edge / max(width, height)
    if scale != 1.0:
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def resolve_cached_frame(
    cache_dir: Path,
    frame_index: int,
    boundary_tolerance_frames: int = 2,
) -> Path | None:
    """Resolve an exact cached frame or a near-boundary decoded frame.

    Args:
        cache_dir: Directory containing zero-padded JPEG frame indexes.
        frame_index: Requested two-fps frame index.
        boundary_tolerance_frames: Maximum allowed clamp distance beyond the
            first or last decoded frame.

    Returns:
        Exact or boundary-clamped JPEG path, or ``None`` for an internal or
        distant gap.
    """
    exact = cache_dir / f"{frame_index:08d}.jpg"
    if exact.exists():
        return exact
    cached = sorted(cache_dir.glob("*.jpg"))
    if not cached:
        return None
    first = cached[0]
    last = cached[-1]
    first_index = int(first.stem)
    last_index = int(last.stem)
    if first_index - boundary_tolerance_frames <= frame_index < first_index:
        return first
    if last_index < frame_index <= last_index + boundary_tolerance_frames:
        return last
    return None


def extract_video_frames(
    video_path: Path,
    timestamp_to_outputs: dict[float, list[Path]],
    long_edge: int,
    jpeg_quality: int,
) -> tuple[int, list[float]]:
    extract_fps = 2
    qscale = 3 if jpeg_quality >= 85 else 4
    frame_to_outputs: dict[int, list[Path]] = defaultdict(list)
    for time_sec, output_paths in timestamp_to_outputs.items():
        output_paths = [output_path for output_path in output_paths if not output_path.exists()]
        if not output_paths:
            continue
        frame_index = max(0, int(round(time_sec * extract_fps)))
        frame_to_outputs[frame_index].extend(output_paths)
    if not frame_to_outputs:
        return 0, []
    first_output = next(iter(frame_to_outputs.values()))[0]
    dataset_root = first_output.parents[2]
    cache_parent = dataset_root / "metadata" / "_ffmpeg_2fps_cache"
    cache_parent.mkdir(parents=True, exist_ok=True)

    extracted = 0
    failed: list[float] = []
    with tempfile.TemporaryDirectory(dir=cache_parent, prefix=f"{video_path.stem}_") as tmp:
        cache_dir = Path(tmp)
        pattern = cache_dir / "%08d.jpg"
        vf = (
            f"fps={extract_fps},"
            f"scale=w='if(gte(iw,ih),{long_edge},-2)':"
            f"h='if(gte(iw,ih),-2,{long_edge})':flags=lanczos"
        )
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-vf",
                vf,
                "-q:v",
                str(qscale),
                "-start_number",
                "0",
                str(pattern),
            ],
            check=True,
        )

        for frame_index, output_paths in sorted(frame_to_outputs.items()):
            source_path = resolve_cached_frame(cache_dir, frame_index)
            if source_path is None:
                failed.append(round(frame_index / extract_fps, 3))
                continue
            for output_path in output_paths:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    continue
                try:
                    os.link(source_path, output_path)
                except OSError:
                    shutil.copy2(source_path, output_path)
                extracted += 1
    return extracted, failed


def extract_video_set(
    video_ids: list[str],
    raw_videos: dict[str, Path],
    timestamp_outputs: dict[str, dict[float, list[Path]]],
    long_edge: int,
    jpeg_quality: int,
    workers: int,
    extractor: Callable[
        [Path, dict[float, list[Path]], int, int],
        tuple[int, list[float]],
    ] = extract_video_frames,
) -> tuple[dict[str, int], dict[str, list[float]]]:
    """Extract multiple source videos concurrently with deterministic results.

    Args:
        video_ids: Ordered source-video IDs to extract.
        raw_videos: Raw source path by video ID.
        timestamp_outputs: Requested timestamps and output paths by video ID.
        long_edge: Stored JPEG long-edge resolution.
        jpeg_quality: Requested JPEG quality.
        workers: Maximum concurrent ffmpeg processes.
        extractor: Per-video extraction implementation.

    Returns:
        Extracted image counts and failed timestamps by video ID.
    """
    worker_count = max(1, int(workers))

    def extract_one(video_id: str) -> tuple[str, int, list[float]]:
        extracted, failed = extractor(
            raw_videos[video_id],
            timestamp_outputs[video_id],
            long_edge,
            jpeg_quality,
        )
        return video_id, extracted, failed

    if worker_count == 1:
        results = [extract_one(video_id) for video_id in video_ids]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(extract_one, video_ids))
    extracted_by_video = {video_id: extracted for video_id, extracted, _ in results}
    failures = {video_id: failed for video_id, _, failed in results if failed}
    return extracted_by_video, failures


def reuse_legacy_images(
    reuse_root: Path,
    output_root: Path,
    raw_videos: dict[str, Path],
    timestamp_outputs: dict[str, dict[float, list[Path]]],
    long_edge: int,
    jpeg_quality: int,
) -> dict:
    """Reuse compatible JPEGs from an older gameplay dataset.

    Frame identity is matched by normalized source-video ID and timestamp, not
    by window ID or class folder. This permits corrected labels to reuse the
    same decoded pixels while new manifests and paths retain the new semantics.
    The source extraction resolution, JPEG setting, and raw-video path must
    agree before any link is created.

    Args:
        reuse_root: Existing extracted gameplay dataset.
        output_root: New dataset being constructed.
        raw_videos: Current raw-video path by normalized video ID.
        timestamp_outputs: Requested new output paths by video and timestamp.
        long_edge: Requested stored JPEG long edge.
        jpeg_quality: Requested JPEG quality setting.

    Returns:
        Reuse counts and link/copy mode details for the dataset summary.

    Raises:
        ValueError: If the source dataset is the output dataset or its
            extraction settings/raw paths are incompatible.
        FileNotFoundError: If required source metadata is missing.
    """
    reuse_root = reuse_root.resolve()
    output_root = output_root.resolve()
    if reuse_root == output_root:
        raise ValueError("--reuse-images-from must differ from --output-root")

    summary_path = reuse_root / "metadata" / "summary.json"
    images_path = reuse_root / "metadata" / "images.jsonl"
    if not summary_path.exists() or not images_path.exists():
        raise FileNotFoundError(
            f"Reuse dataset must contain metadata/summary.json and metadata/images.jsonl: {reuse_root}"
        )
    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_long_edge = source_summary.get("long_edge")
    source_jpeg_quality = source_summary.get("jpeg_quality")
    if source_long_edge != long_edge or source_jpeg_quality != jpeg_quality:
        raise ValueError(
            "Reuse dataset extraction settings differ: "
            f"source long_edge={source_long_edge}, jpeg_quality={source_jpeg_quality}; "
            f"requested long_edge={long_edge}, jpeg_quality={jpeg_quality}"
        )

    requested_keys = {
        (video_id, round(float(time_sec), 3))
        for video_id, times in timestamp_outputs.items()
        for time_sec in times
    }
    reusable: dict[tuple[str, float], Path] = {}
    source_raw_paths: dict[str, Path] = {}
    with images_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id") or normalize_video_id(row.get("source_video", "")))
            key = (video_id, round(float(row["time_sec"]), 3))
            if key not in requested_keys or key in reusable:
                continue
            source_path = reuse_root / row["path"]
            if not source_path.exists():
                continue
            source_raw_path = Path(row["raw_video_path"]).resolve()
            current_raw_path = raw_videos.get(video_id)
            if current_raw_path is None:
                continue
            current_raw_path = current_raw_path.resolve()
            if source_raw_path != current_raw_path:
                raise ValueError(
                    f"Raw-video path differs for reusable source {video_id}: "
                    f"{source_raw_path} != {current_raw_path}"
                )
            source_raw_paths[video_id] = source_raw_path
            reusable[key] = source_path

    reused_by_video: Counter[str] = Counter()
    reused_timestamp_by_video: Counter[str] = Counter()
    hardlink_count = 0
    copied_count = 0
    for video_id, times in timestamp_outputs.items():
        for time_sec, output_paths in times.items():
            source_path = reusable.get((video_id, round(float(time_sec), 3)))
            if source_path is None:
                continue
            reused_this_timestamp = False
            for output_path in output_paths:
                if output_path.exists():
                    continue
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source_path, output_path)
                    hardlink_count += 1
                except OSError:
                    shutil.copy2(source_path, output_path)
                    copied_count += 1
                reused_by_video[video_id] += 1
                reused_this_timestamp = True
            if reused_this_timestamp:
                reused_timestamp_by_video[video_id] += 1

    return {
        "source_dataset": str(reuse_root),
        "source_long_edge": source_long_edge,
        "source_jpeg_quality": source_jpeg_quality,
        "matched_timestamp_count": len(reusable),
        "reused_unique_video_timestamp_count": sum(reused_timestamp_by_video.values()),
        "reused_image_count": sum(reused_by_video.values()),
        "hardlink_count": hardlink_count,
        "copied_count": copied_count,
        "reused_image_count_by_video": dict(sorted(reused_by_video.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_extract(args: argparse.Namespace, extractor: Callable = extract_video_frames) -> None:
    if args.output_root.exists():
        if args.force:
            shutil.rmtree(args.output_root)
        elif not args.resume:
            raise SystemExit(
                f"Output exists: {args.output_root}. Use --resume to preserve it or --force to rebuild."
            )
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)

    subset_video_ids = load_video_manifest(args.video_manifest) if args.video_manifest else None
    labels = read_labels(args.workbook, subset_video_ids)
    regions = build_regions(labels, args.prematch_gameplay_zone_sec, args.other_transition_zone_sec)
    dropped_regions = [region for region in regions if region.stable_duration_sec < args.clip_seconds]
    reference_windows = build_windows(
        regions,
        args.clip_seconds,
        args.gameplay_stride_sec,
        args.negative_stride_sec,
    )
    raw_videos = local_raw_videos(args.raw_root)
    wanted_video_ids = sorted({window["video_id"] for window in reference_windows})
    missing = [video_id for video_id in wanted_video_ids if video_id not in raw_videos]
    if missing:
        raise SystemExit(f"Missing local raw videos for {len(missing)} workbook ids: {', '.join(missing)}")

    available_video_ids = {window["video_id"] for window in reference_windows}
    if args.static_val_metadata:
        base_val_videos = load_static_val_videos(args.static_val_metadata)
        missing_static = sorted(base_val_videos - available_video_ids)
        if missing_static:
            raise SystemExit(
                "Static validation videos from the base dataset are missing in the new workbook: "
                + ", ".join(missing_static)
            )
    elif args.val_video_manifest:
        base_val_videos = load_video_manifest(args.val_video_manifest)
        missing_explicit = sorted(base_val_videos - available_video_ids)
        if missing_explicit:
            raise SystemExit(
                "Explicit validation videos are missing in the selected workbook population: "
                + ", ".join(missing_explicit)
            )
    else:
        base_val_videos = choose_val_videos(reference_windows, args.seed, args.val_fraction)

    expanded_val_videos = (
        load_expanded_val_videos(args.expanded_val_manifest) if args.expanded_val_manifest else set()
    )
    missing_expanded = sorted(expanded_val_videos - available_video_ids)
    if missing_expanded:
        raise SystemExit(
            "Expanded validation videos are missing in the workbook: " + ", ".join(missing_expanded)
        )
    train_negative_stride = args.train_negative_stride_sec or args.negative_stride_sec
    windows = build_staged_windows(
        regions=regions,
        legacy_val_videos=base_val_videos,
        expanded_val_videos=expanded_val_videos,
        clip_seconds=args.clip_seconds,
        train_gameplay_stride=args.gameplay_stride_sec,
        train_negative_stride=train_negative_stride,
        legacy_gameplay_stride=args.gameplay_stride_sec,
        legacy_negative_stride=args.negative_stride_sec,
        expanded_val_stride=args.expanded_val_stride_sec,
        legacy_val_cohort=args.val_cohort_name,
    )
    val_videos = base_val_videos | expanded_val_videos
    image_rows: list[dict] = []
    timestamp_outputs: dict[str, dict[float, list[Path]]] = defaultdict(lambda: defaultdict(list))
    class_counts: Counter[tuple[str, str]] = Counter()
    window_counts: Counter[tuple[str, str]] = Counter()
    cohort_window_counts: Counter[tuple[str, str]] = Counter()

    for window in windows:
        split = args.train_split_name if window["split_population"] == "train" else "val"
        window["split"] = split
        window["boundary_margin_sec"] = max(
            float(window["start_exclude_sec"]),
            float(window["end_exclude_sec"]),
        )
        window["sampling_seed"] = args.seed
        window["raw_video_path"] = str(raw_videos[window["video_id"]])
        frame_paths: list[str] = []
        for frame_index, time_sec in enumerate(window["times_sec"]):
            filename = f"{window['window_id']}_s{frame_index}.jpg"
            path_split = "train" if split in {"train", "trn"} else split
            rel_path = Path(path_split) / window["class_name"] / filename
            output_path = args.output_root / rel_path
            frame_paths.append(str(rel_path))
            timestamp_outputs[window["video_id"]][float(time_sec)].append(output_path)
            image_rows.append(
                {
                    "image_id": f"{window['window_id']}_s{frame_index}",
                    "window_id": window["window_id"],
                    "split": split,
                    "class_name": window["class_name"],
                    "original_category": window["original_category"],
                    "binary_label": window["binary_label"],
                    "source_video": window["source_video"],
                    "video_id": window["video_id"],
                    "validation_cohort": window["validation_cohort"],
                    "raw_video_path": str(raw_videos[window["video_id"]]),
                    "region_start_sec": window["region_start_sec"],
                    "region_end_sec": window["region_end_sec"],
                    "boundary_margin_sec": window["boundary_margin_sec"],
                    "sampling_seed": args.seed,
                    "time_sec": time_sec,
                    "frame_index": frame_index,
                    "path": str(rel_path),
                }
            )
            class_counts[(split, window["class_name"])] += 1
        window["frame_paths"] = frame_paths
        window_counts[(split, window["class_name"])] += 1
        cohort = window["validation_cohort"] or "train"
        cohort_window_counts[(cohort, window["class_name"])] += 1

    reuse_summary = None
    if args.reuse_images_from:
        reuse_summary = reuse_legacy_images(
            reuse_root=args.reuse_images_from,
            output_root=args.output_root,
            raw_videos=raw_videos,
            timestamp_outputs=timestamp_outputs,
            long_edge=args.long_edge,
            jpeg_quality=args.jpeg_quality,
        )

    newly_extracted_by_video, failures = extract_video_set(
        video_ids=wanted_video_ids,
        raw_videos=raw_videos,
        timestamp_outputs=timestamp_outputs,
        long_edge=args.long_edge,
        jpeg_quality=args.jpeg_quality,
        workers=args.extract_workers,
        extractor=extractor,
    )

    if failures:
        raise RuntimeError(f"Frame extraction failures: {failures}")

    expected_image_count_by_video = {
        video_id: sum(len(output_paths) for output_paths in timestamp_outputs[video_id].values())
        for video_id in wanted_video_ids
    }

    metadata_dir = args.output_root / "metadata"
    write_jsonl(metadata_dir / "windows.jsonl", windows)
    write_jsonl(metadata_dir / "images.jsonl", image_rows)
    class_names = sorted({row["class_name"] for row in image_rows})
    (metadata_dir / "class_to_idx.json").write_text(
        json.dumps({name: idx for idx, name in enumerate(class_names)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "workbook": str(args.workbook),
        "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "video_manifest": str(args.video_manifest) if args.video_manifest else None,
        "static_val_metadata": str(args.static_val_metadata) if args.static_val_metadata else None,
        "val_video_manifest": str(args.val_video_manifest) if args.val_video_manifest else None,
        "val_cohort_name": args.val_cohort_name,
        "static_val_video_count": len(base_val_videos),
        "expanded_val_manifest": str(args.expanded_val_manifest) if args.expanded_val_manifest else None,
        "expanded_val_video_count": len(expanded_val_videos),
        "validation_cohort_videos": {
            args.val_cohort_name: sorted(base_val_videos),
            "v5_expanded": sorted(expanded_val_videos),
        },
        "train_split_name": args.train_split_name,
        "val_videos": sorted(val_videos),
        "train_videos": sorted(set(wanted_video_ids) - val_videos),
        "video_count": len(wanted_video_ids),
        "window_count": len(windows),
        "image_count": len(image_rows),
        "unique_video_timestamp_count": sum(len(times) for times in timestamp_outputs.values()),
        "long_edge": args.long_edge,
        "jpeg_quality": args.jpeg_quality,
        "reuse_images_from": str(args.reuse_images_from) if args.reuse_images_from else None,
        "reuse_summary": reuse_summary,
        "extract_workers": args.extract_workers,
        "resumed": args.resume,
        "clip_seconds": args.clip_seconds,
        "region_count": len(regions),
        "dropped_short_region_count": len(dropped_regions),
        "dropped_short_region_counts_by_class": dict(
            sorted(Counter(region.class_dir for region in dropped_regions).items())
        ),
        "dropped_short_region_counts_by_video": dict(
            sorted(Counter(region.video_id for region in dropped_regions).items())
        ),
        "gameplay_stride_sec": args.gameplay_stride_sec,
        "negative_stride_sec": args.negative_stride_sec,
        "train_negative_stride_sec": train_negative_stride,
        "expanded_val_stride_sec": args.expanded_val_stride_sec,
        "sampling_policy": {
            "train": {
                "gameplay_stride_sec": args.gameplay_stride_sec,
                "negative_stride_sec": train_negative_stride,
            },
            args.val_cohort_name: {
                "gameplay_stride_sec": args.gameplay_stride_sec,
                "negative_stride_sec": args.negative_stride_sec,
            },
            "v5_expanded": {
                "gameplay_stride_sec": args.expanded_val_stride_sec,
                "negative_stride_sec": args.expanded_val_stride_sec,
            },
        },
        "prematch_gameplay_zone_sec": args.prematch_gameplay_zone_sec,
        "other_transition_zone_sec": args.other_transition_zone_sec,
        "image_counts_by_split_class": {
            f"{split}/{class_name}": count for (split, class_name), count in sorted(class_counts.items())
        },
        "window_counts_by_split_class": {
            f"{split}/{class_name}": count for (split, class_name), count in sorted(window_counts.items())
        },
        "window_counts_by_cohort_class": {
            f"{cohort}/{class_name}": count
            for (cohort, class_name), count in sorted(cohort_window_counts.items())
        },
        "extracted_image_count_by_video": expected_image_count_by_video,
        "newly_extracted_image_count_by_video": newly_extracted_by_video,
    }
    (metadata_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "sync-raw":
        run_sync_raw(args)
    elif args.command == "validate-raw":
        video_ids = load_video_manifest(args.video_manifest) if args.video_manifest else None
        validate_raw_durations(
            args.workbook,
            args.raw_root,
            args.metadata_dir,
            args.tolerance_sec,
            video_ids,
        )
    elif args.command == "extract":
        run_extract(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
