#!/usr/bin/env python3
"""Extract one v8 raw source with bounded seeks and resumable JPEG writes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from extract_gameplay_pil import balanced_selection, read_ppm


DEFAULT_WORK = Path("/mnt/t/output/skyball_gameplay/v8_import_val50_20261006_r1")
REMOTE_LISTING = Path("/mnt/t/output/skyball_gameplay/v8_reconciliation_20261006/shared_video_listing.json")


def resize_and_save(image: Image.Image, paths: list[Path]) -> int:
    scale = 1024 / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    resized = image.resize(size, Image.Resampling.LANCZOS)
    count = 0
    for path in paths:
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            resized.save(handle, format="JPEG", quality=88)
        count += 1
    return count


def sparse_keyframes(video: Path) -> tuple[bool, list[float]]:
    command = [
        "ffprobe", "-v", "error", "-skip_frame", "nokey", "-read_intervals", "%+180",
        "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=best_effort_timestamp_time,key_frame", "-of", "json", str(video),
    ]
    result = json.loads(subprocess.check_output(command, text=True))
    times = [float(row["best_effort_timestamp_time"]) for row in result["frames"]
             if row.get("key_frame") == 1]
    gaps = [right-left for left, right in zip(times, times[1:])]
    return len(times) < 2 or max(gaps, default=0) > 15, gaps


def extract_sequential(video: Path, needed: dict[int, list[Path]], gpu: int,
                       pixel_format: str) -> int:
    indices = sorted(needed)
    selection = balanced_selection(indices)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2"]
    use_gpu = pixel_format in {"yuv420p", "yuvj420p"}
    if use_gpu:
        command += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu),
                    "-hwaccel_output_format", "cuda"]
    command += ["-i", str(video), "-an", "-sn", "-dn"]
    filter_graph = f"fps=2,select='{selection}'"
    if use_gpu:
        filter_graph += f",hwdownload,format=nv12,format={pixel_format}"
    command += ["-vf", filter_graph, "-fps_mode", "vfr", "-frames:v", str(len(indices)),
                "-threads", "1", "-c:v", "ppm", "-f", "image2pipe", "pipe:1"]
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    count = 0
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env) as process:
        try:
            for index in indices:
                image = read_ppm(process.stdout)
                if image is None:
                    raise RuntimeError(f"Sequential decode ended early at frame {index}")
                count += resize_and_save(image, needed[index])
            process.stdout.close()
            error = process.stderr.read()
            code = process.wait()
            if code:
                raise RuntimeError(error.decode(errors="replace")[-1000:])
        except BaseException:
            process.kill()
            process.wait()
            raise
    return count


def extract_grouped_seeks(video: Path, needed: dict[int, list[Path]]) -> tuple[int, int]:
    groups: list[list[int]] = []
    for index in sorted(needed):
        if not groups or index - groups[-1][-1] > 2:
            groups.append([])
        groups[-1].append(index)
    count = 0
    for group in groups:
        seek = max(0, group[0] / 2 - 1)
        selection = balanced_selection(group).replace("eq(n,", "eq(pts,")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2",
            "-copyts", "-start_at_zero", "-ss", str(seek), "-i", str(video),
            "-an", "-sn", "-dn", "-vf", f"fps=2,select='{selection}'",
            "-fps_mode", "vfr", "-frames:v", str(len(group)), "-threads", "1",
            "-c:v", "ppm", "-f", "image2pipe", "pipe:1",
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            try:
                for index in group:
                    image = read_ppm(process.stdout)
                    if image is None:
                        raise RuntimeError(f"Seek decode ended early at frame {index}")
                    count += resize_and_save(image, needed[index])
                process.stdout.close()
                error = process.stderr.read()
                code = process.wait()
                if code:
                    raise RuntimeError(error.decode(errors="replace")[-1000:])
            except BaseException:
                process.kill()
                process.wait()
                raise
    return count, len(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video_id")
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    started = time.monotonic()
    plan = json.loads((args.work / "plan.json").read_text())
    dataset = Path(plan["dataset"])
    overrides_path = args.work / "raw_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    video = Path(overrides.get(args.video_id,
                               str(Path(plan["raw_root"]) / (args.video_id + ".mp4"))))
    records = [json.loads(line) for line in
               (args.work / "windows_by_video" / f"{args.video_id}.jsonl").open()]
    remote = {row["Name"]: row for row in json.loads(REMOTE_LISTING.read_text())}
    expected_size = int(remote[video.name]["Size"])
    expected_mtime = datetime.fromisoformat(remote[video.name]["ModTime"].replace("Z", "+00:00")).timestamp()
    if (not video.is_file() or video.stat().st_size != expected_size
            or abs(video.stat().st_mtime - expected_mtime) > 1.0):
        raise RuntimeError(f"Raw file absent or still downloading: {video}")
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=index,codec_type,start_time,duration,pix_fmt",
        "-of", "json", str(video),
    ], text=True))
    stream = next(row for row in probe["streams"] if row["codec_type"] == "video")
    duration = float(probe["format"]["duration"])
    visual_end = (float(stream.get("start_time") or 0) + float(stream["duration"])
                  if stream.get("duration") not in (None, "N/A") else duration)
    annotation_end = max(float(row["region_end_sec"]) for row in records)
    if annotation_end > duration + .30:
        raise RuntimeError((args.video_id, "annotations exceed raw duration", annotation_end, duration))
    manifest = json.loads((Path(plan["release"]) / "manifest.json").read_text())
    reviewed = {row["video_id"]: row for row in manifest["model_reviewed_videos"]}
    if args.video_id in reviewed and abs(duration - float(reviewed[args.video_id]["duration_sec"])) > .25:
        raise RuntimeError((args.video_id, "manifest duration differs", duration,
                            reviewed[args.video_id]["duration_sec"]))
    dropped = [row["window_id"] for row in records if float(row["end_sec"]) > visual_end + 1e-6]
    kept = [row for row in records if row["window_id"] not in dropped]
    needed: dict[int, list[Path]] = defaultdict(list)
    for row in kept:
        for stamp, relative in zip(row["decoded_grid_times_sec"], row["frame_paths"], strict=True):
            path = dataset / relative
            if not path.is_file():
                needed[round(float(stamp) * 2)].append(path)
    new_images = 0
    seek_groups = 0
    strategy = "reused"
    keyframe_gaps: list[float] = []
    if needed:
        long_gop, keyframe_gaps = sparse_keyframes(video)
        if long_gop:
            strategy = "sequential_gpu" if stream.get("pix_fmt") in {"yuv420p", "yuvj420p"} else "sequential_cpu"
            new_images = extract_sequential(video, needed, args.gpu, stream.get("pix_fmt", ""))
        else:
            strategy = "grouped_cpu_seeks"
            new_images, seek_groups = extract_grouped_seeks(video, needed)
    for row in kept:
        for relative in row["frame_paths"]:
            path = dataset / relative
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing/empty extracted JPEG: {path}")
    result = {
        "video_id": args.video_id,
        "raw_path": str(video),
        "raw_bytes": video.stat().st_size,
        "duration_sec": duration,
        "visual_end_sec": visual_end,
        "annotation_end_sec": annotation_end,
        "planned_windows": len(records),
        "kept_windows": len(kept),
        "dropped_audio_tail_windows": dropped,
        "new_images": new_images,
        "strategy": strategy,
        "max_probe_keyframe_gap_sec": max(keyframe_gaps, default=None),
        "seek_groups": seek_groups,
        "gpu": args.gpu if strategy == "sequential_gpu" else None,
        "elapsed_sec": round(time.monotonic()-started, 2),
        "status": "complete",
    }
    out = args.work / "video_checks"
    out.mkdir(exist_ok=True)
    dest = out / f"{args.video_id}.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
