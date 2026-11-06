#!/usr/bin/env python3
"""One 1920x1080 four-frame GIF per missed GT event (what the rally net saw).

Frames: the 1 Hz window whose centre (start + 1.5 s) is nearest the event, i.e. the
four frames at start .. start+3 s. Each frame is captioned with the source, event,
frame offset from the event and that window's rally score.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skyball_gameplay.data_root import data_root  # noqa: E402

IMAGES = data_root() / 'images_full'
SUITE = Path('/mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1')
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)


def window_scores_cuts(predictions):
    out = {}
    for r in csv.DictReader(open(predictions)):
        start = int(re.search(r'_t(\d{10})$', r['window_id']).group(1)) / 1000
        out.setdefault(Path(r['source_video']).stem, {})[start] = float(r['score_gameplay'])
    return out


def window_scores_full10(score_dir, match):
    return {float(r['window_start_sec']): float(r['score_gameplay'])
            for r in csv.DictReader(open(Path(score_dir) / match / 'scores.csv'))}


def frame_cut(cut, sec):
    return Image.open(IMAGES / cut / f'{round(sec * 30) + 1:06d}.jpg').convert('RGB').resize((1920, 1080))


def frame_video(video, sec):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{sec:.3f}', '-i', str(video), '-frames:v', '1',
                          '-vf', 'scale=1920:1080', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         check=True, capture_output=True).stdout
    return Image.frombytes('RGB', (1920, 1080), raw)


def render(frames, captions, path):
    out = []
    for im, cap in zip(frames, captions):
        d = ImageDraw.Draw(im)
        for i, line in enumerate(cap):
            y = 12 + 40 * i
            d.text((14, y), line, font=FONT, fill=(255, 255, 0), stroke_width=3, stroke_fill=(0, 0, 0))
        out.append(im.convert('P', palette=Image.ADAPTIVE, colors=256))
    out[0].save(path, save_all=True, append_images=out[1:], duration=900, loop=0, optimize=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True, help='evaluate_rally.py JSON')
    p.add_argument('--name', required=True, help='score-set name in the report')
    p.add_argument('--policy', default='0.5:0.0')
    p.add_argument('--scores', required=True, help='cuts: predictions.csv; full10: score dir')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    res = report['results'][args.name]['fixed_policies'][args.policy]
    args.output.mkdir(parents=True, exist_ok=True)
    cuts = report['source'] == 'cuts'
    scores_all = window_scores_cuts(args.scores) if cuts else None
    index = []
    for v in res['per_video']:
        scores = scores_all[v['video']] if cuts else window_scores_full10(args.scores, v['video'])
        video = None if cuts else Path(json.loads((SUITE / v['video'] / 'manifest.json').read_text())['video'])
        for m in v['missed_events']:
            t = m['sec']
            start = max(0, min(max(scores), math.floor(t - 1.5 + .5)))
            score = scores.get(float(start), float('nan'))
            frames, captions = [], []
            for k in range(4):
                s = start + k
                frames.append(frame_cut(v['video'], s) if cuts else frame_video(video, s))
                captions.append([f"{v['video']}  MISSED: {m['label']} @ {t:.2f}s",
                                 f"frame {k + 1}/4  t={s:.0f}s ({s - t:+.2f}s from event)  "
                                 f"window rally score {score:.3f}"])
            name = f"{args.name}_{v['video']}_{t:08.2f}s_{re.sub(r'[^A-Za-z0-9]+', '_', m['label']).strip('_')}.gif"
            render(frames, captions, args.output / name)
            index.append({'gif': name, 'video': v['video'], 'event_sec': t, 'label': m['label'],
                          'window_start_sec': start, 'window_score': score})
            print(name, flush=True)
    (args.output / f'{args.name}_index.json').write_text(json.dumps(index, indent=2) + '\n')


if __name__ == '__main__':
    main()
