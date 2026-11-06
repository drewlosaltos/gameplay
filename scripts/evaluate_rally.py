#!/usr/bin/env python3
"""Rally evaluation that puts missed GT touches first (2026-11-05 study).

Truth: a frame is rally when it lies within ``--margin-sec`` (2.0) of a GT event.
Sources:
  * ``full10``: canonical ten matches, confirmed synchronized GT, score dirs holding
    ``<match>/scores.csv`` (window_start_sec, window_end_sec, score_gameplay).
  * ``cuts``: held-out sparse v4 cuts, human events, a ``predictions.csv`` written by
    ``skyball_gameplay.evaluation --save-predictions`` on the rally dataset.

Metrics per score set:
  1. Admission policy sweep (threshold x padding), prod convention: a window at or above
     threshold admits [start - pad, end + pad]. For each policy: GT events missed (event
     frame not admitted), dead-time rejection (fraction of non-rally frames not admitted)
     and admitted share. Selected policy = max dead-time rejection with zero missed
     events; if none has zero, fewest misses then max rejection.
  2. Centre-timestamp frame F1 (window score painted on centre +/- 0.5 s): strict, and
     border-tolerant (frames within ``--tolerance-sec`` of a truth edge are ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SUITE = Path('/mnt/t/output/skyball_pipeline/prod_xl_square_latestpipe_full10_20260818_v1')
PSEUDO = Path('/mnt/t/data/vball/skyball/unified_dataset/derived/detector/pseudo/sparse_v4_xl_dense_fullcls_'
              'ctx12_active6_stable_ballactor_h085_low035_deduptrackids_human_events')
THRESHOLDS = [round(x, 2) for x in np.arange(.05, .96, .05)] + [.97, .98, .99]
PADDINGS = [0, 1, 2, 3, 5, 8, 10, 15, 20]


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def load_full10(score_dir):
    from skyball_gameplay.full10 import validate_full10_inputs
    videos = []
    for mp in validate_full10_inputs(SUITE):
        man = json.loads(mp.read_text())
        timing = json.loads((mp.parent / 'gameplay_timing.json').read_text())
        events = [r for r in read_csv(man['gt_sync_csv']) if r.get('status', '').strip().lower() == 'confirmed']
        rows = [(float(r['window_start_sec']), float(r['window_end_sec']), float(r['score_gameplay']))
                for r in read_csv(Path(score_dir) / mp.parent.name / 'scores.csv')]
        videos.append({'name': mp.parent.name, 'count': int(timing['source_frames']), 'fps': float(man['fps']),
                       'events': [int(round(float(r['synced_fnum']))) for r in events],
                       'event_labels': [f"set{r['set']} rally{r['rally']} {r.get('skill', r.get('action', ''))}"
                                        for r in events], 'rows': rows})
    return videos


def load_cuts(predictions):
    by_cut = {}
    for r in read_csv(predictions):
        cut = Path(r['source_video']).stem
        start = int(re.search(r'_t(\d{10})$', r['window_id']).group(1)) / 1000
        by_cut.setdefault(cut, []).append((start, start + 4., float(r['score_gameplay'])))
    videos = []
    for cut, rows in sorted(by_cut.items()):
        meta = json.loads((PSEUDO / cut / 'metadata.json').read_text())
        ev = json.loads((PSEUDO / cut / 'events.json').read_text())['events']
        videos.append({'name': cut, 'count': int(meta['num_frames']), 'fps': 30.,
                       'events': [int(e['frame']) for e in ev],
                       'event_labels': [f"{e['skill']} {e.get('side', '')}" for e in ev], 'rows': rows})
    return videos


def truth_mask(v, margin):
    m = np.zeros(v['count'], bool)
    r = int(round(margin * v['fps']))
    for f in v['events']:
        m[max(0, f - 1 - r):min(v['count'], f + r)] = True
    return m


def edge_band(truth, fps, tol):
    """Frames within tol seconds of a truth transition."""
    band = np.zeros_like(truth)
    edges = np.flatnonzero(np.diff(truth.astype(np.int8))) + 1
    r = int(round(tol * fps))
    for e in edges:
        band[max(0, e - r):e + r] = True
    return band


def policy_mask(v, th, pad, post=None):
    post = pad if post is None else post
    m = np.zeros(v['count'], bool)
    fps = v['fps']
    for a, b, s in v['rows']:
        if s >= th:
            m[max(0, math.floor((a - pad) * fps)):min(v['count'], math.ceil((b + post) * fps))] = True
    return m


def centre_trace(v):
    s = np.zeros(v['count'], np.float32)
    for a, _, score in v['rows']:
        c = a + 1.5
        s[max(0, math.floor((c - .5) * v['fps'])):min(v['count'], math.ceil((c + .5) * v['fps']))] = score
    return s


def prf(t, p):
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    pr = tp / (tp + fp) if tp + fp else 0.
    rc = tp / (tp + fn) if tp + fn else 0.
    return {'precision': pr, 'recall': rc, 'f1': 2 * pr * rc / (pr + rc) if pr + rc else 0.}


def evaluate(videos, margin, tol, fixed_policies):
    truths = [truth_mask(v, margin) for v in videos]
    bands = [edge_band(t, v['fps'], tol) for t, v in zip(truths, videos)]
    total_events = sum(len(v['events']) for v in videos)
    dead = sum(int((~t).sum()) for t in truths)
    frames = sum(v['count'] for v in videos)

    def run_policy(th, pad, detail=False, post=None):
        missed, rejected, admitted, out = 0, 0, 0, []
        for v, t in zip(videos, truths):
            m = policy_mask(v, th, pad, post)
            miss = [(f, lab) for f, lab in zip(v['events'], v['event_labels']) if not m[f - 1]]
            missed += len(miss); rejected += int((~t & ~m).sum()); admitted += int(m.sum())
            if detail:
                out.append({'video': v['name'], 'events': len(v['events']), 'missed': len(miss),
                            'missed_events': [{'frame': f, 'sec': round((f - 1) / v['fps'], 2), 'label': lab}
                                              for f, lab in miss],
                            'dead_time_rejection': float((~t & ~m).sum() / max(1, (~t).sum())),
                            'admitted_pct': 100 * float(m.mean())})
        res = {'threshold': th, 'padding_sec': pad, 'post_padding_sec': pad if post is None else post,
               'events_missed': missed,
               'events_missed_pct': 100 * missed / total_events,
               'dead_time_rejection': rejected / dead, 'admitted_pct': 100 * admitted / frames}
        if detail:
            res['per_video'] = out
        return res

    sweep = [run_policy(th, pad) for th in THRESHOLDS for pad in PADDINGS]
    # Pre-roll only: extend admitted spans earlier (serves start rallies), little or no tail padding.
    sweep += [run_policy(th, pre, post=post) for th in THRESHOLDS for pre in PADDINGS for post in (0, 1, 2)
              if pre != post]
    best = min(sweep, key=lambda r: (r['events_missed'], -r['dead_time_rejection']))
    selected = run_policy(best['threshold'], best['padding_sec'], detail=True, post=best['post_padding_sec'])
    fixed = {':'.join(map(str, spec)): run_policy(spec[0], spec[1], detail=True,
                                                  post=spec[2] if len(spec) > 2 else None)
             for spec in fixed_policies}
    t = np.concatenate(truths); s = np.concatenate([centre_trace(v) for v in videos])
    keep = ~np.concatenate(bands)
    centre = {f'@{th}': {'strict': prf(t, s >= th), 'border_tolerant': prf(t[keep], s[keep] >= th)}
              for th in (.5, .9)}
    best_tol = max(THRESHOLDS, key=lambda th: prf(t[keep], s[keep] >= th)['f1'])
    centre['best_border_tolerant'] = {'threshold': best_tol, **prf(t[keep], s[keep] >= best_tol)}
    return {'videos': len(videos), 'events': total_events, 'frames': frames,
            'truth_rally_pct': 100 * float(t.mean()), 'selected_policy': selected,
            'fixed_policies': fixed, 'centre_frame_metrics': centre,
            'zero_miss_frontier': sorted([r for r in sweep if r['events_missed'] == 0],
                                         key=lambda r: -r['dead_time_rejection'])[:10],
            'sweep': sweep}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', choices=['full10', 'cuts'], required=True)
    p.add_argument('--scores', action='append', required=True,
                   help='name=path (full10: score dir; cuts: predictions.csv)')
    p.add_argument('--policy', action='append', default=[], help='threshold:padding or threshold:pre:post (seconds)')
    p.add_argument('--margin-sec', type=float, default=2.0)
    p.add_argument('--tolerance-sec', type=float, default=1.0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    fixed = [tuple(float(x) for x in s.split(':')) for s in args.policy]
    report = {'source': args.source, 'margin_sec': args.margin_sec, 'tolerance_sec': args.tolerance_sec,
              'definition': __doc__, 'results': {}}
    for spec in args.scores:
        name, path = spec.split('=', 1)
        videos = load_full10(path) if args.source == 'full10' else load_cuts(path)
        r = evaluate(videos, args.margin_sec, args.tolerance_sec, fixed)
        r['path'] = path
        report['results'][name] = r
        sel, c = r['selected_policy'], r['centre_frame_metrics']
        print(f"{name}: {r['videos']} videos {r['events']} events, truth rally {r['truth_rally_pct']:.1f}%")
        print(f"  selected th {sel['threshold']} pre {sel['padding_sec']}s post {sel['post_padding_sec']}s: missed {sel['events_missed']}/{r['events']} ({sel['events_missed_pct']:.2f}%)  "
              f"dead-time rejected {100 * sel['dead_time_rejection']:.1f}%  admitted {sel['admitted_pct']:.1f}%")
        for k, fp in r['fixed_policies'].items():
            print(f"  policy {k}: missed {fp['events_missed']}/{r['events']} ({fp['events_missed_pct']:.2f}%)  dead-time rejected "
                  f"{100 * fp['dead_time_rejection']:.1f}%  admitted {fp['admitted_pct']:.1f}%")
        print(f"  centre F1@0.5 strict {c['@0.5']['strict']['f1']:.4f}  border-tolerant "
              f"{c['@0.5']['border_tolerant']['f1']:.4f}  (P {c['@0.5']['border_tolerant']['precision']:.3f} "
              f"R {c['@0.5']['border_tolerant']['recall']:.3f})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
