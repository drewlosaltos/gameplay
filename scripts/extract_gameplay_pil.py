#!/usr/bin/env python3
"""Use the existing gameplay sampler with selective decoding and PIL JPEG output.

Retains the legacy two-Hz frame selection (nearest half second), while resizing
selected RGB frames with PIL Lanczos and encoding at the requested JPEG quality.
No temporary full-video frame cache is needed. All existing images are preserved.
"""

from __future__ import annotations

import os
import subprocess
from collections import defaultdict
from pathlib import Path

from PIL import Image

import build_gameplay_v0_dataset as builder


def read_ppm(stream):
    header = stream.readline()
    if not header:
        return None
    if header.strip() != b"P6":
        raise ValueError(f"Unexpected ffmpeg PPM header: {header!r}")
    width, height = map(int, stream.readline().split())
    if stream.readline().strip() != b"255":
        raise ValueError("Expected eight-bit RGB output")
    remaining = width * height * 3
    chunks = []
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise ValueError("Truncated decoded RGB frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return Image.frombytes("RGB", (width, height), b"".join(chunks))


def balanced_selection(indices):
    """Keep FFmpeg expression recursion logarithmic for thousands of frames."""
    if len(indices) == 1:
        return f"eq(n,{indices[0]})"
    middle = len(indices) // 2
    return f"({balanced_selection(indices[:middle])}+{balanced_selection(indices[middle:])})"


def extract_video_frames(video_path, timestamp_to_outputs, long_edge, jpeg_quality):
    frames = defaultdict(list)
    for timestamp, paths in timestamp_to_outputs.items():
        for path in paths:
            if not path.exists():
                frames[max(0, round(timestamp * 2))].append(path)
    if not frames:
        return 0, []
    indices = sorted(frames)
    selection = balanced_selection(indices)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2"]
    if os.environ.get("GAMEPLAY_HWACCEL"):
        command += ["-hwaccel", os.environ["GAMEPLAY_HWACCEL"]]
    command += [
        "-i", str(video_path), "-an", "-sn", "-dn",
        "-vf", f"fps=2,select='{selection}'", "-fps_mode", "vfr",
        "-frames:v", str(len(indices)), "-threads", "1",
        "-c:v", "ppm", "-f", "image2pipe", "pipe:1",
    ]
    count = 0
    completed = set()
    print(f"EXTRACT_START {video_path.name} frames={len(indices)}", flush=True)
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        try:
            for index in indices:
                image = read_ppm(process.stdout)
                if image is None:
                    break
                width, height = image.size
                scale = long_edge / max(width, height)
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                image = image.resize(size, Image.Resampling.LANCZOS)
                for path in frames[index]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open("xb") as handle:
                        image.save(handle, format="JPEG", quality=jpeg_quality)
                    count += 1
                completed.add(index)
            process.stdout.close()
            if process.wait():
                raise RuntimeError(f"ffmpeg failed for {video_path}")
        except BaseException:
            process.kill()
            process.wait()
            raise
    failures = [index / 2 for index in indices if index not in completed]
    print(f"EXTRACT_DONE {video_path.name} images={count} failures={len(failures)}", flush=True)
    return count, failures


if __name__ == "__main__":
    args = builder.parse_args()
    if args.command != "extract":
        raise SystemExit("This entry point supports only the extract subcommand")
    builder.run_extract(args, extractor=extract_video_frames)
