#!/usr/bin/env python3
"""Sync raw SkyBall gameplay videos and build an ImageNet-style frame dataset."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image

try:
    import ffmpegcv
except ImportError:  # pragma: no cover - environment dependent
    ffmpegcv = None


DEFAULT_WORKBOOK = Path("/mnt/t/data/vball/skyball/unified_staging/gameplay.xlsx")
DEFAULT_RAW_ROOT = Path("/mnt/t/data/vball/skyball/raw_video")
DEFAULT_OUTPUT_ROOT = Path("/mnt/t/data/vball/skyball/gameplay/v0")
DEFAULT_REMOTE = "googdrive:skyball/raw_video"

CLASS_DIRS = {
    "gameplay": "gameplay",
    "timeout": "timeout",
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
    sync.add_argument("--dry-run", action="store_true")

    extract = subparsers.add_parser("extract", help="extract sampled 4-frame windows into ImageNet folders")
    extract.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    extract.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    extract.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    extract.add_argument("--force", action="store_true")
    extract.add_argument("--seed", type=int, default=20260628)
    extract.add_argument("--val-fraction", type=float, default=0.20)
    extract.add_argument("--clip-seconds", type=int, default=4)
    extract.add_argument("--gameplay-stride-sec", type=float, default=20.0)
    extract.add_argument("--negative-stride-sec", type=float, default=1.0)
    extract.add_argument("--prematch-gameplay-zone-sec", type=float, default=20.0)
    extract.add_argument("--other-transition-zone-sec", type=float, default=5.0)
    extract.add_argument("--long-edge", type=int, default=512)
    extract.add_argument("--jpeg-quality", type=int, default=88)
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


def read_labels(workbook: Path) -> pd.DataFrame:
    labels = pd.read_excel(workbook)
    labels = labels.dropna(subset=["video", "category", "start_sec", "end_sec"]).copy()
    labels["source_video"] = labels["video"].astype(str).str.strip()
    labels["video_id"] = labels["source_video"].map(normalize_video_id)
    labels["category"] = labels["category"].astype(str).str.strip()
    labels["category_norm"] = labels["category"].map(normalize_category)
    labels["class_dir"] = labels["category"].map(class_dir_for)
    labels["start_sec"] = labels["start_sec"].astype(float)
    labels["end_sec"] = labels["end_sec"].astype(float)
    labels = labels.sort_values(["video_id", "start_sec", "end_sec"]).reset_index(drop=False)
    labels = labels.rename(columns={"index": "row_index"})
    return labels


def workbook_video_names(workbook: Path) -> set[str]:
    labels = read_labels(workbook)
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
    wanted = workbook_video_names(args.workbook)
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


def resize_lanczos(image_bgr, long_edge: int) -> Image.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    width, height = image.size
    scale = long_edge / max(width, height)
    if scale != 1.0:
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


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
            source_path = cache_dir / f"{frame_index:08d}.jpg"
            if not source_path.exists():
                found = None
                for delta in (-1, 1):
                    candidate = cache_dir / f"{frame_index + delta:08d}.jpg"
                    if candidate.exists():
                        found = candidate
                        break
                if found is None:
                    failed.append(round(frame_index / extract_fps, 3))
                    continue
                source_path = found
            for output_path in output_paths:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if output_path.exists():
                    output_path.unlink()
                try:
                    os.link(source_path, output_path)
                except OSError:
                    shutil.copy2(source_path, output_path)
                extracted += 1
    return extracted, failed


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def run_extract(args: argparse.Namespace) -> None:
    if args.output_root.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {args.output_root}. Use --force to rebuild.")
        shutil.rmtree(args.output_root)
    (args.output_root / "metadata").mkdir(parents=True, exist_ok=True)

    labels = read_labels(args.workbook)
    regions = build_regions(labels, args.prematch_gameplay_zone_sec, args.other_transition_zone_sec)
    windows = build_windows(regions, args.clip_seconds, args.gameplay_stride_sec, args.negative_stride_sec)
    raw_videos = local_raw_videos(args.raw_root)
    wanted_video_ids = sorted({window["video_id"] for window in windows})
    missing = [video_id for video_id in wanted_video_ids if video_id not in raw_videos]
    if missing:
        raise SystemExit(f"Missing local raw videos for {len(missing)} workbook ids: {', '.join(missing)}")

    val_videos = choose_val_videos(windows, args.seed, args.val_fraction)
    image_rows: list[dict] = []
    timestamp_outputs: dict[str, dict[float, list[Path]]] = defaultdict(lambda: defaultdict(list))
    class_counts: Counter[tuple[str, str]] = Counter()
    window_counts: Counter[tuple[str, str]] = Counter()

    for window in windows:
        split = "val" if window["video_id"] in val_videos else "trn"
        window["split"] = split
        window["raw_video_path"] = str(raw_videos[window["video_id"]])
        frame_paths: list[str] = []
        for frame_index, time_sec in enumerate(window["times_sec"]):
            filename = f"{window['window_id']}_s{frame_index}.jpg"
            rel_path = Path(split) / window["class_name"] / filename
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
                    "raw_video_path": str(raw_videos[window["video_id"]]),
                    "time_sec": time_sec,
                    "frame_index": frame_index,
                    "path": str(rel_path),
                }
            )
            class_counts[(split, window["class_name"])] += 1
        window["frame_paths"] = frame_paths
        window_counts[(split, window["class_name"])] += 1

    extracted_by_video = {}
    failures: dict[str, list[float]] = {}
    for video_id in wanted_video_ids:
        extracted, failed = extract_video_frames(
            raw_videos[video_id],
            timestamp_outputs[video_id],
            args.long_edge,
            args.jpeg_quality,
        )
        extracted_by_video[video_id] = extracted
        if failed:
            failures[video_id] = failed

    if failures:
        raise RuntimeError(f"Frame extraction failures: {failures}")

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
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "val_videos": sorted(val_videos),
        "train_videos": sorted(set(wanted_video_ids) - val_videos),
        "video_count": len(wanted_video_ids),
        "window_count": len(windows),
        "image_count": len(image_rows),
        "unique_video_timestamp_count": sum(len(times) for times in timestamp_outputs.values()),
        "long_edge": args.long_edge,
        "jpeg_quality": args.jpeg_quality,
        "clip_seconds": args.clip_seconds,
        "gameplay_stride_sec": args.gameplay_stride_sec,
        "negative_stride_sec": args.negative_stride_sec,
        "prematch_gameplay_zone_sec": args.prematch_gameplay_zone_sec,
        "other_transition_zone_sec": args.other_transition_zone_sec,
        "image_counts_by_split_class": {
            f"{split}/{class_name}": count for (split, class_name), count in sorted(class_counts.items())
        },
        "window_counts_by_split_class": {
            f"{split}/{class_name}": count for (split, class_name), count in sorted(window_counts.items())
        },
        "extracted_image_count_by_video": extracted_by_video,
    }
    (metadata_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.command == "sync-raw":
        run_sync_raw(args)
    elif args.command == "extract":
        run_extract(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
