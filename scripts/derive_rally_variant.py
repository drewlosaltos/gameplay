#!/usr/bin/env python3
"""Derive a training-time variant of a rally dataset; validation rows are copied unchanged.

Rallies are segmented per cut from the human sparse-v4 events: a new rally starts at a
serve or after a gap of more than ``--rally-gap-sec`` (5 s). Every event in a rally
counts toward its length, including floor_bounce / ball_in_net / net_touch;
``--exclude-skills`` drops skills from the count (short4x_20261106_r1 was built with
``--exclude-skills floor_bounce,ball_in_net,net_touch``).

Modes (train split only):
  * ``oversample``: rally windows whose nearest event belongs to a short rally
    (<= ``--short-max-touches`` events, e.g. serve-only or serve + floor bounce) are
    repeated ``--factor`` times in total.
  * ``serve_pre``: windows whose centre lies in [serve - ``--serve-pre-sec``, serve) are
    relabelled rally, for the serve that opens each rally (serve formation and toss).

Frames are shared with the base dataset through a ``frames`` symlink.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PSEUDO = Path('/mnt/t/data/vball/skyball/unified_dataset/derived/detector/pseudo/sparse_v4_xl_dense_fullcls_'
              'ctx12_active6_stable_ballactor_h085_low035_deduptrackids_human_events')
FPS = 30.0
SERVES = {'serve', 'serve_invisible'}


def rallies(cut: str, gap: float) -> list[list[dict]]:
    events = sorted(json.loads((PSEUDO / cut / 'events.json').read_text())['events'], key=lambda e: e['frame'])
    out, cur = [], []
    for e in events:
        if cur and (e['skill'] in SERVES or (e['frame'] - cur[-1]['frame']) / FPS > gap):
            out.append(cur)
            cur = []
        cur.append(e)
    if cur:
        out.append(cur)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['oversample', 'serve_pre'], required=True)
    p.add_argument('--rally-gap-sec', type=float, default=5.0)
    p.add_argument('--short-max-touches', type=int, default=2)
    p.add_argument('--exclude-skills', default='', help='comma-separated skills not counted toward rally length')
    p.add_argument('--factor', type=int, default=4)
    p.add_argument('--serve-pre-sec', type=float, default=4.0)
    args = p.parse_args()
    skip = {k for k in args.exclude_skills.split(',') if k}
    if args.output.exists():
        p.error('Use a new versioned output directory')
    base_rows = [json.loads(line) for line in (args.base / 'metadata/windows.jsonl').open()]
    cache, rows, stats = {}, [], {'train_rows_added': 0, 'train_relabelled': 0, 'short_rallies': 0, 'rallies': 0}
    for w in base_rows:
        if w['split'] not in ('train', 'trn'):
            rows.append(w)
            continue
        cut = w['video_id']
        if cut not in cache:
            rs = rallies(cut, args.rally_gap_sec)
            cache[cut] = rs
            stats['rallies'] += len(rs)
            stats['short_rallies'] += sum(len([e for e in r if e['skill'] not in skip]) <= args.short_max_touches
                                          for r in rs)
        rs, c = cache[cut], w['center_sec']
        if args.mode == 'oversample':
            rows.append(w)
            if w['binary_label']:
                nearest = min(rs, key=lambda r: min(abs(c - (e['frame'] - 1) / FPS) for e in r))
                if len([e for e in nearest if e['skill'] not in skip]) <= args.short_max_touches:
                    rows += [dict(w, oversample_copy=k) for k in range(1, args.factor)]
                    stats['train_rows_added'] += args.factor - 1
        else:
            serve_starts = [(r[0]['frame'] - 1) / FPS for r in rs if r[0]['skill'] in SERVES]
            if not w['binary_label'] and any(t - args.serve_pre_sec <= c < t for t in serve_starts):
                w = dict(w, binary_label=1, class_name='rally',
                         label_policy=w['label_policy'] + f'+serve_pre_{args.serve_pre_sec}s')
                stats['train_relabelled'] += 1
            rows.append(w)
    (args.output / 'metadata').mkdir(parents=True)
    (args.output / 'frames').symlink_to((args.base / 'frames').resolve())
    text = ''.join(json.dumps(w, sort_keys=True) + '\n' for w in rows)
    (args.output / 'metadata/windows.jsonl').write_text(text)
    (args.output / 'metadata/class_to_idx.json').write_text((args.base / 'metadata/class_to_idx.json').read_text())
    counts = {}
    for w in rows:
        key = f"{w['split']}_{w['class_name']}"
        counts[key] = counts.get(key, 0) + 1
    summary = {'base': str(args.base.resolve()), 'mode': args.mode, 'args': {k: str(v) for k, v in vars(args).items()},
               'stats': stats, 'window_counts': counts, 'windows_sha256': hashlib.sha256(text.encode()).hexdigest()}
    (args.output / 'metadata/summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary['stats']), json.dumps(counts))


if __name__ == '__main__':
    main()
