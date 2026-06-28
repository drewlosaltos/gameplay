#!/usr/bin/env python3
"""Build annotated gameplay review clips from SkyBall gameplay labels."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd


DEFAULT_WORKBOOK = Path("/mnt/t/data/vball/skyball/unified_staging/gameplay.xlsx")
DEFAULT_IMAGE_ROOT = Path("/mnt/t/data/vball/skyball/unified_dataset/images_full")
DEFAULT_RAW_VIDEO_ROOT = Path("/mnt/t/data/vball/skyball/raw_video")
DEFAULT_OUTPUT_ROOT = Path("/mnt/t/output/skyball_gameplay/review_clips")

CLASS_PRIORITY = [
    "gameplay",
    "warmup",
    "pre-match",
    "pre_match",
    "prematch",
    "set break",
    "set_break",
    "timeout",
    "stoppage",
    "post-match",
    "post_match",
    "ceremony",
]


@dataclass(frozen=True)
class FrameIndex:
    video: str
    duration_sec: float
    frame_paths: tuple[Path, ...]

    @property
    def image_rate(self) -> float:
        return len(self.frame_paths) / self.duration_sec if self.duration_sec > 0 else 0.0

    def path_for_time(self, time_sec: float) -> Path:
        if not self.frame_paths:
            raise ValueError(f"No frames for {self.video}")
        if len(self.frame_paths) == 1 or self.duration_sec <= 0:
            return self.frame_paths[0]
        pos = min(max(time_sec / self.duration_sec, 0.0), 1.0) * (len(self.frame_paths) - 1)
        return self.frame_paths[int(round(pos))]


@dataclass(frozen=True)
class SourceVideo:
    video: str
    path: Path


@dataclass(frozen=True)
class Region:
    row_index: int
    source_video: str
    category: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class Sample:
    sample_id: str
    video: str
    source_video: str
    category: str
    start_sec: float
    end_sec: float
    source_region_start_sec: float
    source_region_end_sec: float
    source_row_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["dense", "sparse"],
        default="dense",
        help="dense renders from raw match videos; sparse renders from images_full play clips",
    )
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--raw-video-root", type=Path, default=DEFAULT_RAW_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="review_dense_v0")
    parser.add_argument("--videos", nargs="*", help="Workbook video ids to render")
    parser.add_argument("--num-videos", type=int, default=4)
    parser.add_argument("--samples-per-video", type=int, default=12)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--boundary-margin-sec", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--allow-sparse",
        action="store_true",
        help="allow frame-folder rendering; not valid for raw-match gameplay.xlsx timestamp review",
    )
    parser.add_argument("--encoder", default="h264_nvenc", choices=["h264_nvenc", "libx264"])
    return parser.parse_args()


def hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    return f"{whole // 3600:02d}:{(whole % 3600) // 60:02d}:{whole % 60:02d}"


def normalize_video_id(value: object) -> str:
    return Path(str(value).strip()).stem


def normalize_category(value: object) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def play_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_play(\d+)$", path.name)
    return (int(match.group(1)) if match else 9999, path.name)


def build_frame_indices(image_root: Path, labels: pd.DataFrame) -> dict[str, FrameIndex]:
    indices: dict[str, FrameIndex] = {}
    for video, group in labels.groupby("video", sort=True):
        frame_paths: list[Path] = []
        for frame_dir in sorted(image_root.glob(f"{video}_play*"), key=play_sort_key):
            frame_paths.extend(sorted(frame_dir.glob("*.jpg")))
        if not frame_paths:
            continue
        duration_sec = float(group["end_sec"].max())
        rel_paths = tuple(path.relative_to(image_root) for path in frame_paths)
        indices[video] = FrameIndex(video=video, duration_sec=duration_sec, frame_paths=rel_paths)
    return indices


def build_source_videos(raw_video_root: Path, labels: pd.DataFrame) -> dict[str, SourceVideo]:
    wanted = {str(video) for video in labels["video"].dropna().unique()}
    sources: dict[str, SourceVideo] = {}
    for path in raw_video_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
            continue
        if path.stem in wanted and path.stem not in sources:
            sources[path.stem] = SourceVideo(video=path.stem, path=path)
    return sources


def eligible_regions(
    labels: pd.DataFrame,
    available_videos: set[str],
    clip_seconds: float,
    boundary_margin_sec: float,
) -> dict[str, list[Region]]:
    by_video: dict[str, list[Region]] = {}
    for video, group in labels.groupby("video", sort=True):
        if video not in available_videos:
            continue
        regions: list[Region] = []
        for row_index, row in group.iterrows():
            start = float(row["start_sec"])
            end = float(row["end_sec"])
            stable_start = start + boundary_margin_sec
            stable_end = end - boundary_margin_sec
            if stable_end - stable_start < clip_seconds:
                continue
            regions.append(
                Region(
                    row_index=int(row_index),
                    source_video=str(row.get("source_video", video)),
                    category=str(row["category"]),
                    start_sec=stable_start,
                    end_sec=stable_end,
                )
            )
        if regions:
            by_video[video] = regions
    return by_video


def category_key(category: str) -> tuple[int, str]:
    category = category.strip().lower()
    try:
        return (CLASS_PRIORITY.index(category), category)
    except ValueError:
        return (len(CLASS_PRIORITY), category)


def choose_videos(
    eligible: dict[str, list[Region]],
    explicit_videos: Iterable[str] | None,
    num_videos: int,
) -> list[str]:
    if explicit_videos:
        explicit_videos = [normalize_video_id(video) for video in explicit_videos]
        missing = [video for video in explicit_videos if video not in eligible]
        if missing:
            raise SystemExit(f"No eligible regions for requested videos: {', '.join(missing)}")
        return list(explicit_videos)

    def score(video: str) -> tuple[int, int, float, str]:
        cats = {region.category for region in eligible[video]}
        return (len(cats), len(eligible[video]), 0.0, video)

    return sorted(eligible, key=score, reverse=True)[:num_videos]


def choose_samples(
    video: str,
    regions: list[Region],
    samples_per_video: int,
    clip_seconds: float,
    rng: random.Random,
) -> list[Sample]:
    by_category: dict[str, list[Region]] = {}
    for region in regions:
        by_category.setdefault(region.category, []).append(region)

    categories = sorted(by_category, key=category_key)
    samples: list[Sample] = []
    attempts = 0
    while len(samples) < samples_per_video and attempts < samples_per_video * max(4, len(categories)):
        category = categories[attempts % len(categories)]
        region = rng.choice(by_category[category])
        start = rng.uniform(region.start_sec, region.end_sec - clip_seconds)
        sample_id = f"{video}_{len(samples):02d}_{category}_{int(round(start * 1000)):09d}"
        samples.append(
            Sample(
                sample_id=sample_id,
                video=video,
                source_video=region.source_video,
                category=category,
                start_sec=start,
                end_sec=start + clip_seconds,
                source_region_start_sec=region.start_sec,
                source_region_end_sec=region.end_sec,
                source_row_index=region.row_index,
            )
        )
        attempts += 1
    return samples


def draw_overlay(frame, lines: list[str]):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(frame.shape[1] / 1280.0, 0.7)
    thickness = max(2, int(round(scale * 2)))
    line_height = int(round(30 * scale))
    padding = int(round(14 * scale))
    box_height = padding * 2 + line_height * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    y = padding + int(round(22 * scale))
    for line in lines:
        cv2.putText(frame, line, (padding, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height
    return frame


def render_sample_frames(
    sample: Sample,
    frame_index: FrameIndex,
    image_root: Path,
    frame_dir: Path,
    fps: int,
    width: int,
) -> list[str]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, int(round((sample.end_sec - sample.start_sec) * fps)))
    used_paths: list[str] = []
    last_image = None
    for out_idx in range(frame_count):
        offset = out_idx / fps
        video_time = sample.start_sec + offset
        rel_path = frame_index.path_for_time(video_time)
        used_paths.append(str(rel_path))
        image = cv2.imread(str(image_root / rel_path))
        if image is None:
            if last_image is None:
                raise RuntimeError(f"Could not read {image_root / rel_path}")
            image = last_image.copy()
        else:
            last_image = image
        if width and image.shape[1] != width:
            height = int(round(image.shape[0] * (width / image.shape[1])))
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        lines = [
            f"{sample.source_video}    {sample.category}    {hms(video_time)}",
            f"sample {sample.sample_id}    source row {sample.source_row_index}",
        ]
        image = draw_overlay(image, lines)
        cv2.imwrite(str(frame_dir / f"{out_idx:06d}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return used_paths


def run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def encode_clip(frame_dir: Path, output_path: Path, fps: int, encoder: str) -> None:
    common = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%06d.jpg"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if encoder == "h264_nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    try:
        run_ffmpeg(common + codec_args + [str(output_path)])
    except subprocess.CalledProcessError:
        if encoder != "h264_nvenc":
            raise
        run_ffmpeg(common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(output_path)])


def escape_drawtext(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    for old, new in {
        ":": "\\:",
        "'": "\\'",
        "%": "\\%",
        ",": "\\,",
        "[": "\\[",
        "]": "\\]",
    }.items():
        escaped = escaped.replace(old, new)
    return escaped


def dense_filter(sample: Sample, width: int) -> str:
    line1 = f"{sample.source_video}    {sample.category}    {hms(sample.start_sec)}-{hms(sample.end_sec)}"
    line2 = f"sample {sample.sample_id}    source row {sample.source_row_index}"
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    filters = []
    if width:
        filters.append(f"scale=w={width}:h=-2")
    filters.extend(
        [
            "drawbox=x=0:y=0:w=iw:h=108:color=black@0.62:t=fill",
            (
                f"drawtext=fontfile='{font}':text='{escape_drawtext(line1)}':"
                "x=24:y=18:fontsize=30:fontcolor=white"
            ),
            (
                f"drawtext=fontfile='{font}':text='{escape_drawtext(line2)}':"
                "x=24:y=62:fontsize=24:fontcolor=white"
            ),
        ]
    )
    return ",".join(filters)


def render_dense_sample(
    sample: Sample,
    source_video: SourceVideo,
    output_path: Path,
    width: int,
    encoder: str,
) -> None:
    common = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{sample.start_sec:.3f}",
        "-i",
        str(source_video.path),
        "-t",
        f"{sample.end_sec - sample.start_sec:.3f}",
        "-vf",
        dense_filter(sample, width),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    if encoder == "h264_nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    try:
        run_ffmpeg(common + codec_args + [str(output_path)])
    except subprocess.CalledProcessError:
        if encoder != "h264_nvenc":
            raise
        run_ffmpeg(common + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(output_path)])


def concat_clips(sample_paths: list[Path], output_path: Path, list_path: Path) -> None:
    with list_path.open("w", encoding="utf-8") as handle:
        for path in sample_paths:
            handle.write(f"file '{path.resolve()}'\n")
    run_ffmpeg(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(output_path),
        ]
    )


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode == "sparse" and not args.allow_sparse:
        raise SystemExit(
            "Sparse/images_full rendering is not valid for gameplay.xlsx raw-match timestamps. "
            "Use --mode dense with raw videos, or pass --allow-sparse only for diagnostics."
        )
    rng = random.Random(args.seed)

    output_dir = args.output_root / args.run_name
    if output_dir.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {output_dir}. Use --force or a new --run-name.")
        shutil.rmtree(output_dir)
    sample_dir = output_dir / "samples"
    combined_dir = output_dir / "combined"
    tmp_root = output_dir / "_frames_tmp"
    sample_dir.mkdir(parents=True)
    combined_dir.mkdir(parents=True)

    labels = pd.read_excel(args.workbook)
    labels = labels.dropna(subset=["video", "category", "start_sec", "end_sec"]).copy()
    labels["source_video"] = labels["video"].astype(str).str.strip()
    labels["video"] = labels["source_video"].map(normalize_video_id)
    labels["category"] = labels["category"].astype(str).str.strip()
    frame_indices: dict[str, FrameIndex] = {}
    source_videos: dict[str, SourceVideo] = {}
    if args.mode == "dense":
        source_videos = build_source_videos(args.raw_video_root, labels)
        available_videos = set(source_videos)
    else:
        frame_indices = build_frame_indices(args.image_root, labels)
        available_videos = set(frame_indices)
    eligible = eligible_regions(labels, available_videos, args.clip_seconds, args.boundary_margin_sec)
    videos = choose_videos(eligible, args.videos, args.num_videos)

    records = []
    with tempfile.TemporaryDirectory(dir=output_dir, prefix="_tmp_frames_") as tmp:
        tmp_base = Path(tmp)
        for video in videos:
            samples = choose_samples(video, eligible[video], args.samples_per_video, args.clip_seconds, rng)
            sample_paths = []
            for sample in samples:
                sample_path = sample_dir / f"{sample.sample_id}.mp4"
                record = {
                    "sample_id": sample.sample_id,
                    "source_video": sample.source_video,
                    "video_id": sample.video,
                    "category": sample.category,
                    "binary_label": 1 if normalize_category(sample.category) == "gameplay" else 0,
                    "start_sec": round(sample.start_sec, 3),
                    "end_sec": round(sample.end_sec, 3),
                    "start_hms": hms(sample.start_sec),
                    "end_hms": hms(sample.end_sec),
                    "source_region_start_sec": round(sample.source_region_start_sec, 3),
                    "source_region_end_sec": round(sample.source_region_end_sec, 3),
                    "source_row_index": sample.source_row_index,
                    "render_mode": args.mode,
                }
                if args.mode == "dense":
                    render_dense_sample(sample, source_videos[video], sample_path, args.width, args.encoder)
                    record["source_video_path"] = str(source_videos[video].path)
                    record["frame_source"] = "raw_video"
                else:
                    frame_index = frame_indices[video]
                    frame_dir = tmp_base / sample.sample_id
                    used_paths = render_sample_frames(sample, frame_index, args.image_root, frame_dir, args.fps, args.width)
                    encode_clip(frame_dir, sample_path, args.fps, args.encoder)
                    record.update(
                        {
                            "image_root": str(args.image_root),
                            "frame_source": "images_full",
                            "used_frame_paths": used_paths,
                            "estimated_image_rate": frame_index.image_rate,
                            "frame_time_mapping": "linear_over_concatenated_sorted_play_folders",
                        }
                    )
                sample_paths.append(sample_path)
                records.append(record)
            concat_clips(sample_paths, combined_dir / f"{video}_review.mp4", output_dir / f"{video}_concat.txt")

    manifest_jsonl = output_dir / "review_manifest.jsonl"
    with manifest_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "workbook": str(args.workbook),
        "raw_video_root": str(args.raw_video_root),
        "output_dir": str(output_dir),
        "render_mode": args.mode,
        "frame_source": "raw_video" if args.mode == "dense" else "images_full",
        "seed": args.seed,
        "boundary_margin_sec": args.boundary_margin_sec,
        "clip_seconds": args.clip_seconds,
        "videos": videos,
        "combined_videos": [str(combined_dir / f"{video}_review.mp4") for video in videos],
        "sample_count": len(records),
        "category_counts": pd.Series([r["category"] for r in records]).value_counts().to_dict(),
        "mapped_video_count": len(frame_indices),
        "source_video_count": len(source_videos),
        "eligible_video_count": len(eligible),
    }
    if args.mode == "sparse":
        summary["image_root"] = str(args.image_root)
        summary["fps"] = args.fps
        summary["frame_time_mapping"] = "linear_over_concatenated_sorted_play_folders"
    else:
        summary["output_fps"] = "source_native"
        summary["source_video_paths"] = {video: str(source_videos[video].path) for video in videos}
    write_json(output_dir / "summary.json", summary)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
