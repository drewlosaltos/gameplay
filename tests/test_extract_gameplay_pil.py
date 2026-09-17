"""Integration checks for decoded timing, image geometry, and existing files."""

import importlib.util
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("extract_gameplay_pil", SCRIPTS / "extract_gameplay_pil.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selected_frames_keep_timing_geometry_and_existing_images(tmp_path):
    video = tmp_path / "colors.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=red:s=160x80:r=10:d=1",
        "-f", "lavfi", "-i", "color=blue:s=160x80:r=10:d=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "libx264", str(video),
    ], check=True)
    red, blue, preserved, missing = [tmp_path / (s + ".jpg") for s in ["red", "blue", "preserved", "missing"]]
    preserved.write_bytes(b"existing image must not be replaced")
    timestamps = {0.0: [red], 1.0: [blue], 0.5: [preserved], 5.0: [missing]}
    # Match-sized selections must not exceed FFmpeg's expression recursion limit.
    timestamps.update({float(t): [tmp_path / f"late_{t}.jpg"] for t in range(6, 1500)})
    count, failures = MODULE.extract_video_frames(video, timestamps, 128, 88)
    assert count == 2
    assert failures == [float(t) for t in range(5, 1500)]
    assert preserved.read_bytes() == b"existing image must not be replaced"
    assert not missing.exists()
    for path, channel in [(red, 0), (blue, 2)]:
        with Image.open(path) as image:
            assert image.size == (128, 64)
            rgb = image.getpixel((64, 32))
            assert rgb[channel] > 240
            assert all(value < 15 for i, value in enumerate(rgb) if i != channel)
