#!/usr/bin/env python3
"""Build a four-frame rally/no-rally window dataset from sparse v4 human events.

Label policy (2026-11-05 study): a window of frames at t, t+1, t+2, t+3 seconds is
``rally`` when its center t+1.5 s lies within ``--margin-sec`` (2.0) of any human
sparse-v4 event in that cut; otherwise ``no_rally``. Frames come from the cut's
``images_full`` JPEGs (event frame numbers index the same files), sampled at 1 Hz,
resized to 1024 px on the long edge with Lanczos, JPEG quality 88. Train/val are
split by cut. Only cuts whose ``events.json`` carries human events are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skyball_gameplay.data_root import data_root, dataset_path  # noqa: E402

DATA = data_root()
PSEUDO = DATA / ('derived/detector/pseudo/sparse_v4_xl_dense_fullcls_ctx12_active6_stable_'
                 'ballactor_h085_low035_deduptrackids_human_events')
FPS = 30.0
LONG_EDGE = 1024


def human_cuts() -> list[str]:
    cuts = []
    for d in sorted(PSEUDO.glob('*_cut')):
        if json.loads((d / 'events.json').read_text()).get('source'):
            cuts.append(d.name)
    return cuts


def extract_cut(job: tuple[str, str, Path]) -> dict:
    from PIL import Image
    cut, split, out_root = job
    meta = json.loads((PSEUDO / cut / 'metadata.json').read_text())
    events = json.loads((PSEUDO / cut / 'events.json').read_text())['events']
    count = int(meta['num_frames'])
    event_frames = sorted(int(e['frame']) for e in events)
    event_sec = [(f - 1) / FPS for f in event_frames]
    seconds = int((count - 1) / FPS) + 1  # sample s such that frame round(s*FPS)+1 <= count
    seconds = [s for s in range(seconds) if round(s * FPS) + 1 <= count]
    rel_dir = Path('frames') / cut
    (out_root / rel_dir).mkdir(parents=True, exist_ok=True)
    frame_paths = {}
    for s in seconds:
        frame = round(s * FPS) + 1
        rel = rel_dir / f'{cut}_t{s * 1000:010d}.jpg'
        dst = out_root / rel
        if not dst.exists():
            with Image.open(DATA / 'images_full' / cut / f'{frame:06d}.jpg') as im:
                im = im.convert('RGB')
                scale = LONG_EDGE / max(im.size)
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
                tmp = dst.with_suffix('.tmp.jpg')
                im.save(tmp, quality=88)
                tmp.rename(dst)
        frame_paths[s] = str(rel)

    def nearest(t: float) -> float:
        return min((abs(t - e) for e in event_sec), default=float('inf'))

    windows = []
    for s in seconds:
        if s + 3 not in frame_paths:
            break
        center = s + 1.5
        dist = nearest(center)
        label = int(dist <= MARGIN)
        windows.append({
            'window_id': f'{cut}_w{s:06d}_t{s * 1000:010d}',
            'source_video': f'{cut}.mp4', 'video_id': cut, 'split': split,
            'frame_paths': [frame_paths[s + k] for k in range(4)],
            'times_sec': [float(s + k) for k in range(4)],
            'source_frames': [round((s + k) * FPS) + 1 for k in range(4)],
            'center_sec': center, 'nearest_event_sec': round(dist, 4),
            'frame_rally_labels': [int(nearest(s + k) <= MARGIN) for k in range(4)],
            # GT events whose frame lies inside this window's span [s, s+4) seconds; the
            # admission convention used for "GT events missed" (window start..end, no padding).
            'covered_event_ids': [f'{cut}:{f}' for f in event_frames
                                  if round(s * FPS) <= f - 1 < round((s + 4) * FPS)],
            'class_name': 'rally' if label else 'no_rally', 'binary_label': label,
            'label_policy': f'center_within_{MARGIN}s_of_human_event',
            'annotation_provenance': 'sparse_v4_human_events',
        })
    covered = {e for w in windows for e in w['covered_event_ids']}
    return {'cut': cut, 'split': split, 'frames': count, 'events': len(event_sec),
            'events_uncovered_by_windows': len(event_frames) - len(covered),
            'sampled_seconds': len(seconds), 'windows': windows}


def main() -> None:
    global MARGIN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True,
                   help='new versioned directory; a relative path goes under the data root, '
                        'e.g. derived/gameplay/rally_sparse_v4_m2_<date>_r1')
    p.add_argument('--margin-sec', type=float, default=2.0)
    p.add_argument('--val-cuts', type=int, default=12)
    p.add_argument('--seed', type=int, default=20261105)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    MARGIN = args.margin_sec
    args.output = dataset_path(args.output)
    if (args.output / 'metadata/windows.jsonl').exists():
        p.error('Output already complete; use a new versioned directory')
    cuts = human_cuts()
    val = set(random.Random(args.seed).sample(cuts, args.val_cuts))
    (args.output / 'metadata').mkdir(parents=True, exist_ok=True)
    jobs = [(c, 'val' if c in val else 'train', args.output) for c in cuts]
    results = []
    with ProcessPoolExecutor(args.workers, initializer=_set_margin, initargs=(MARGIN,)) as pool:
        for i, r in enumerate(pool.map(extract_cut, jobs), 1):
            results.append(r)
            (args.output / 'build_status.json').write_text(json.dumps({'done': i, 'total': len(jobs)}))
    rows = [w for r in results for w in r['windows']]
    text = ''.join(json.dumps(w, sort_keys=True) + '\n' for w in rows)
    (args.output / 'metadata/windows.jsonl').write_text(text)
    counts = {}
    for w in rows:
        key = f"{w['split']}_{w['class_name']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {
        'label_policy': f'window center (t+1.5 s) within {MARGIN} s of a human sparse-v4 event',
        'fps_assumed': FPS, 'long_edge': LONG_EDGE, 'seed': args.seed,
        'source': str(PSEUDO), 'val_cuts': sorted(val), 'train_cuts': sorted(set(cuts) - val),
        'window_counts': counts, 'windows_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'per_cut': [{k: r[k] for k in ('cut', 'split', 'frames', 'events', 'events_uncovered_by_windows',
                                    'sampled_seconds')} |
                    {'windows': len(r['windows']),
                     'rally_fraction': round(sum(w['binary_label'] for w in r['windows']) /
                                             max(1, len(r['windows'])), 4)} for r in results],
    }
    (args.output / 'metadata/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output / 'metadata/class_to_idx.json').write_text(json.dumps({'no_rally': 0, 'rally': 1}) + '\n')
    print(json.dumps(counts))


MARGIN = 2.0


def _set_margin(value: float) -> None:
    global MARGIN
    MARGIN = value


if __name__ == '__main__':
    main()
