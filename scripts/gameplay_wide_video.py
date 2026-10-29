"""Pipeline-equivalent 1 Hz sampling with candidate-specific decode dimensions."""
import subprocess
import numpy as np
from skyball_pipeline.gameplay_runtime import probe_video, scaled_dimensions, read_exact


def iter_wide_rgb_frames(video, *, sample_fps, duration_limit_sec, long_edge):
    info = probe_video(video)
    width, height = scaled_dimensions(int(info['width']), int(info['height']), long_edge=long_edge)
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(video), '-an']
    if duration_limit_sec > 0:
        command += ['-t', f'{duration_limit_sec:.9f}']
    command += ['-vf', f'fps=fps={sample_fps}:round=near,scale={width}:{height}:flags=bilinear',
                '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    count = width*height*3
    try:
        while True:
            payload = read_exact(process.stdout, count)
            if not payload:
                break
            if len(payload) != count:
                raise RuntimeError('Truncated candidate gameplay frame')
            yield np.frombuffer(payload, dtype=np.uint8).reshape(height,width,3).copy()
        error = process.stderr.read().decode('utf-8', errors='replace')
        if process.wait():
            raise RuntimeError(error)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
