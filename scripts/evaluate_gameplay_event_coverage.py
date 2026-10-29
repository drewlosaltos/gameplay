#!/usr/bin/env python3
"""Audit gameplay admission against confirmed events and rally envelopes.

This is a read-only retrospective analysis. Complements of event/rally masks
are proxy negatives, not independently annotated non-gameplay.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from skyball_gameplay.full10 import validate_full10_inputs
from scripts.evaluate_gameplay_gate import predicted_intervals


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_mask(ranges, count):
    """Union one-based inclusive ranges; clip only to video boundaries."""
    mask = np.zeros(count, dtype=bool)
    for start, end in ranges:
        if end < start:
            raise ValueError('Reversed frame range')
        first, last = max(1, start), min(count, end)
        if first <= last:
            mask[first - 1:last] = True
    return mask


def positive_ranges(scores, fps, count, threshold, padding, merge_gap_sec=0.):
    """Convert the selected temporal policy to exact source-frame ranges."""
    if merge_gap_sec:
        starts = [float(row['window_start_sec']) for row in scores]
        values = [float(row['score_gameplay']) for row in scores]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite score')
        window_sizes = {round(float(row['window_end_sec']) - float(row['window_start_sec']), 6)
                        for row in scores}
        if len(window_sizes) != 1:
            raise ValueError(f'Inconsistent scored-window durations: {window_sizes}')
        windows = predicted_intervals(starts, values, threshold, window_sizes.pop(),
                                      merge_gap_sec, padding, padding, count / fps)
        return [(max(1, math.floor(interval.start_sec * fps) + 1),
                 min(count, math.ceil(interval.end_sec * fps)))
                for interval in windows]
    ranges = []
    for row in scores:
        score = float(row['score_gameplay'])
        if not math.isfinite(score):
            raise ValueError('Nonfinite score')
        if score >= threshold:
            start = max(0., float(row['window_start_sec']) - padding)
            end = float(row['window_end_sec']) + padding
            first = max(1, math.floor(start * fps) + 1)
            last = min(count, math.ceil(end * fps))
            if first <= last:
                ranges.append((first, last))
    return ranges


def ground_truth_masks(events, count, fps, margin_sec):
    """Build event unions and full first-to-last-event rally envelopes."""
    margin = int(round(margin_sec * fps))
    grouped = defaultdict(list)
    event_ranges = []
    for row in events:
        frame = int(round(float(row['synced_fnum'])))
        if not 1 <= frame <= count:
            raise ValueError(f'GT frame outside video: {frame}/{count}')
        event_ranges.append((max(1, frame - margin), min(count, frame + margin)))
        if not row.get('set') or not row.get('rally'):
            raise ValueError('Missing GT rally identity')
        grouped[(row['set'], row['rally'])].append(frame)
    rally_ranges = [(max(1, min(v) - margin), min(count, max(v) + margin))
                    for v in grouped.values()]
    return frame_mask(event_ranges, count), frame_mask(rally_ranges, count), event_ranges, rally_ranges


def confusion(truth, prediction):
    if truth.shape != prediction.shape:
        raise ValueError('Frame populations differ')
    return {'tp': int(np.sum(truth & prediction)),
            'fn': int(np.sum(truth & ~prediction)),
            'fp': int(np.sum(~truth & prediction)),
            'tn': int(np.sum(~truth & ~prediction))}


def rates(c):
    tp, fn, fp, tn = (c[k] for k in ['tp', 'fn', 'fp', 'tn'])
    divide = lambda a, b: a / b if b else None
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    return {'gameplay_recall': recall, 'miss_rate': divide(fn, tp + fn),
            'outside_region_admission_rate': divide(fp, tn + fp),
            'outside_region_rejection_rate': specificity,
            'proxy_accuracy': divide(tp + tn, tp + fn + fp + tn),
            'proxy_balanced_accuracy': (recall + specificity) / 2
                if recall is not None and specificity is not None else None,
            'proxy_precision': divide(tp, tp + fp),
            'proxy_f1': divide(2 * tp, 2 * tp + fp + fn)}


def true_runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip((np.flatnonzero(changes == 1) + 1).tolist(),
                    np.flatnonzero(changes == -1).tolist()))


def evaluate_prediction(mask, events, count, fps, margin_sec):
    event_mask, rally_mask, event_ranges, rally_ranges = ground_truth_masks(events, count, fps, margin_sec)
    result = {'frames': count, 'fps': fps, 'duration_seconds': count / fps,
              'admitted_frames': int(mask.sum()), 'admitted_seconds': float(mask.sum()) / fps,
              'events': len(events), 'rallies': len(rally_ranges)}
    for name, truth in [('event_margin', event_mask), ('rally_envelope', rally_mask)]:
        c = confusion(truth, mask)
        result[name] = {'frames': c, 'seconds': {k: v / fps for k, v in c.items()}, **rates(c),
                        'missed_intervals': [
                            {'first_frame': a, 'last_frame': b,
                             'start_sec': (a - 1) / fps, 'end_sec': b / fps,
                             'duration_sec': (b - a + 1) / fps}
                            for a, b in true_runs(truth & ~mask)]}
    result['excluded_events'] = [row for row in events
        if not mask[int(round(float(row['synced_fnum']))) - 1]]
    result['events_excluded'] = len(result['excluded_events'])
    result['outside_event_margin_admitted_pct_match'] = 100 * result['event_margin']['frames']['fp'] / count
    result['event_windows_fully_preserved'] = sum(bool(mask[a-1:b].all()) for a, b in event_ranges)
    result['rallies_fully_preserved'] = sum(bool(mask[a-1:b].all()) for a, b in rally_ranges)
    return result


def aggregate(rows):
    totals = {k: sum(row[k] for row in rows) for k in
              ['frames', 'duration_seconds', 'admitted_frames', 'admitted_seconds',
               'events', 'events_excluded', 'rallies', 'rallies_fully_preserved',
               'event_windows_fully_preserved']}
    for definition in ['event_margin', 'rally_envelope']:
        totals[definition] = {}
        for unit in ['frames', 'seconds']:
            c = {k: sum(row[definition][unit][k] for row in rows)
                 for k in ['tp', 'fn', 'fp', 'tn']}
            totals[definition][unit] = c
            totals[definition][unit + '_rates'] = rates(c)
    totals['outside_event_margin_admitted_pct_match'] = 100 * totals['event_margin']['seconds']['fp'] / totals['duration_seconds']
    return totals


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pipeline-root', type=Path, required=True)
    p.add_argument('--candidate-scores', type=Path)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--margin-sec', type=float, default=3.)
    p.add_argument('--candidate-threshold', type=float, default=.15)
    p.add_argument('--candidate-padding-sec', type=float, default=10.)
    p.add_argument('--candidate-merge-gap-sec', type=float, default=0.)
    args = p.parse_args()
    if args.margin_sec != 3.:
        p.error('The canonical contract uses a three-second margin')
    manifests = validate_full10_inputs(args.pipeline_root)
    if (not 0 <= args.candidate_threshold <= 1
            or not math.isfinite(args.candidate_padding_sec) or args.candidate_padding_sec < 0
            or not math.isfinite(args.candidate_merge_gap_sec) or args.candidate_merge_gap_sec < 0):
        p.error('Threshold must be in [0,1], padding and merge gap nonnegative')
    args.output_root.mkdir(parents=True, exist_ok=False)
    first_timing = json.loads((manifests[0].parent/'gameplay_timing.json').read_text())
    policies = {'production_saved': {k: first_timing[k] for k in ['threshold','padding_sec','merge_gap_sec']}}
    if policies['production_saved']['merge_gap_sec'] != 0:
        raise ValueError('This implementation requires a zero-gap baseline policy')
    if args.candidate_scores:
        policies['production_selected_policy'] = {'threshold': args.candidate_threshold, 'padding_sec': args.candidate_padding_sec, 'merge_gap_sec': args.candidate_merge_gap_sec}
        policies.update(candidate_production_policy=dict(policies['production_saved']),
                        candidate_selected_policy={'threshold': args.candidate_threshold, 'padding_sec': args.candidate_padding_sec, 'merge_gap_sec': args.candidate_merge_gap_sec})
    report = {'contract_id': 'gameplay_event_and_rally_frame_coverage_v1',
        'source_pipeline_root': str(args.pipeline_root), 'margin_sec': args.margin_sec,
        'definition': 'Confirmed synced_fnum +/- round(margin_sec * video FPS), inclusive, unioned and clipped. Rally envelope spans first to last confirmed event in each (set,rally), with same margin.',
        'caveat': 'Frames outside annotated event/rally regions are proxy negatives, not verified non-gameplay. Policies frozen before this analysis; candidate inference uses PyTorch FP32 with production preprocessing, not a deployed TensorRT export.',
        'policies': policies, 'per_match': [], 'provenance': []}
    candidate_checkpoint_sha = None
    for path in manifests:
        match = path.parent.name
        manifest = json.loads(path.read_text())
        timing = json.loads((path.parent / 'gameplay_timing.json').read_text())
        count, fps = int(timing['source_frames']), float(manifest['fps'])
        assert manifest['run_status'] == 'succeeded'
        for field in ['threshold', 'padding_sec', 'merge_gap_sec']:
            assert timing[field] == policies['production_saved'][field]
        gt = Path(manifest['gt_sync_csv'])
        assert sha256(gt) == manifest['inputs']['gt_sync_csv']['sha256']
        for key in ['scores_csv', 'timing_json']:
            artifact = manifest['gameplay_runtime_outputs'][key]
            assert sha256(artifact['path']) == artifact['sha256']
        all_gt = read_csv(gt)
        events = [row for row in all_gt if row.get('status', '').strip().lower() == 'confirmed']
        assert events and all(row.get('synced_fnum') for row in events)
        # The run's GT export provides a second check on the scoring population.
        exported = read_csv(path.parent / 'gt_actions.csv')
        assert sorted(int(row['fnum']) for row in exported) == sorted(int(round(float(row['synced_fnum']))) for row in events)
        stored_ranges = [(int(row['fnum-start']), int(row['fnum-last']))
                         for row in read_csv(path.parent / 'gameplay.csv')
                         if int(row['fnum-start']) > 0]
        baseline = frame_mask(stored_ranges, count)
        scores = read_csv(path.parent / 'gameplay_scores.csv')
        rebuilt = frame_mask(positive_ranges(scores, fps, count, timing['threshold'], timing['padding_sec']), count)
        assert np.array_equal(baseline, rebuilt), f'Production range reconstruction differs: {match}'
        masks = {'production_saved': baseline}
        if args.candidate_scores:
            masks['production_selected_policy'] = frame_mask(positive_ranges(scores, fps, count, args.candidate_threshold, args.candidate_padding_sec, args.candidate_merge_gap_sec), count)
        provenance = {'match': match, 'source_manifest': str(path), 'gt': str(gt), 'gt_sha256': sha256(gt),
                      'gameplay_csv_sha256': sha256(path.parent/'gameplay.csv'),
                      'video': manifest['video'], 'video_resolved': str(Path(manifest['video']).resolve()),
                      'production_ranges_reconstructed_exactly': True, 'gt_export_frame_population_verified': True}
        if args.candidate_scores:
            folder = args.candidate_scores / match
            meta = json.loads((folder / 'metadata.json').read_text())
            if candidate_checkpoint_sha is None:
                candidate_checkpoint_sha = meta['checkpoint_sha256']
            if meta['checkpoint_sha256'] != candidate_checkpoint_sha:
                raise ValueError('Candidate scores mix different checkpoints')
            assert Path(meta['video']).resolve() == Path(manifest['video']).resolve()
            assert meta['gt_sha256'] == sha256(gt)
            assert meta['video_info']['frame_count'] == count
            provenance['candidate_metadata'] = meta
            provenance['candidate_scores_sha256'] = sha256(folder/'scores.csv')
            candidate = read_csv(folder / 'scores.csv')
            assert len(candidate) == len(scores)
            assert all(a['window_start_sec'] == b['window_start_sec'] and a['window_end_sec'] == b['window_end_sec'] for a,b in zip(candidate,scores))
            for name, policy in policies.items():
                if name.startswith('candidate_'):
                    masks[name] = frame_mask(positive_ranges(candidate, fps, count, policy['threshold'], policy['padding_sec'], policy['merge_gap_sec']), count)
        row = {'match': match, 'models': {name: evaluate_prediction(mask, events, count, fps, args.margin_sec)
                                        for name, mask in masks.items()}}
        report['per_match'].append(row)
        report['provenance'].append(provenance)
    report['aggregate'] = {name: aggregate([row['models'][name] for row in report['per_match']]) for name in policies}
    for value in report['aggregate'].values():
        if value['events'] != 8848 or value['rallies'] != 1370:
            raise ValueError('Canonical event/rally population changed')
    (args.output_root/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    csv_rows=[]
    for row in report['per_match']:
        for name, m in row['models'].items():
            csv_rows.append({'match':row['match'],'model':name,'events_excluded':m['events_excluded'],
                'outside_event_margin_admitted_pct_match':m['outside_event_margin_admitted_pct_match'],
                'events':m['events'],'rallies_preserved':m['rallies_fully_preserved'],'rallies':m['rallies'],
                **{f'{definition}_{k}':v for definition in ['event_margin','rally_envelope'] for k,v in m[definition]['seconds'].items()},
                'admitted_seconds':m['admitted_seconds']})
    with (args.output_root/'per_match.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps(report['aggregate'],indent=2))


if __name__ == '__main__':
    main()
