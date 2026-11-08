#!/usr/bin/env python3
"""Finish candidate full10 scores and compare them with verified current production."""
import json
import runpy
from pathlib import Path

workflow = runpy.run_path('/mnt/t/output/skyball_gameplay/v8_workflow_20261006/validate_single_stage_when_ready.py')
root = workflow['ROOT']
candidate = workflow['CANDIDATE']
production = workflow['PROD']
candidate_sha = workflow['sha256'](candidate)
production_sha = workflow['sha256'](production)
rows = workflow['score_full10'](candidate_sha)
view = workflow['make_score_view'](rows)
reports = workflow['gate'](candidate_sha, view)

candidate_coverage = reports['candidate_coverage']
production_coverage = reports['production_coverage']
candidate_impact = reports['candidate_impact']
production_impact = reports['production_impact']

def per_match_coverage(report):
    return {row['match']: row['models']['candidate_selected_policy'] for row in report['per_match']}

def per_match_impact(report):
    return {row['match_id']: row for row in report['per_match']}

cc, pc = per_match_coverage(candidate_coverage), per_match_coverage(production_coverage)
ci, pi = per_match_impact(candidate_impact), per_match_impact(production_impact)
score_meta = {row['match']: row for row in rows}
match_rows = []
for match in sorted(cc):
    c, p = cc[match], pc[match]
    cm, pm = ci[match], pi[match]
    match_rows.append({
        'match': match,
        'events': c['events'],
        'production_events_excluded': p['events_excluded'],
        'candidate_events_excluded': c['events_excluded'],
        'production_event_recall': p['event_margin']['gameplay_recall'],
        'candidate_event_recall': c['event_margin']['gameplay_recall'],
        'production_admitted_minutes': p['admitted_seconds'] / 60,
        'candidate_admitted_minutes': c['admitted_seconds'] / 60,
        'production_extra_footage_pct_match': p['outside_event_margin_admitted_pct_match'],
        'candidate_extra_footage_pct_match': c['outside_event_margin_admitted_pct_match'],
        'production_target_fp_rejected': pm['outside_gt_event_margin_action_false_positives']['rejected'],
        'candidate_target_fp_rejected': cm['outside_gt_event_margin_action_false_positives']['rejected'],
        'production_target_fp_remaining': pm['outside_gt_event_margin_action_false_positives']['remaining'],
        'candidate_target_fp_remaining': cm['outside_gt_event_margin_action_false_positives']['remaining'],
        'inference_runtime_sec': score_meta[match]['elapsed_sec'],
    })

summary = {
    'status': 'complete',
    'candidate': {'checkpoint': str(candidate), 'sha256': candidate_sha},
    'production': {'checkpoint': str(production), 'sha256': production_sha,
                   'cached_scores_verified_at': str(workflow['WORK'] / 'prod_full10_cache_verified.json')},
    'canonical_full10_root': str(workflow['PRIMARY']),
    'pipeline_suite_root': str(workflow['SUITE']),
    'policy': {'threshold': 0.5, 'padding_sec_per_side': 20.0, 'merge_gap_sec': 0.0},
    'comparison_basis': 'Candidate scored on all ten canonical inputs; production score cache verified for exact checkpoint, videos, and synced GT. Identical temporal policy.',
    'candidate_full10_runtime_sec_sum': sum(row['elapsed_sec'] for row in rows),
    'candidate_coverage_aggregate': candidate_coverage['aggregate']['candidate_selected_policy'],
    'production_coverage_aggregate': production_coverage['aggregate']['candidate_selected_policy'],
    'candidate_impact_aggregate': candidate_impact['aggregate'],
    'production_impact_aggregate': production_impact['aggregate'],
    'candidate_contract': reports['candidate_contract'],
    'production_contract': json.loads((workflow['WORK'] / 'prod_current_impact_050_20/validation_gate_result.json').read_text()),
    'per_match': match_rows,
    'evidence': {
        'candidate_scores': str(root / 'full10_scores'),
        'production_scores': str(workflow['PROD_SCORE']),
        'candidate_coverage': str(root / 'candidate_coverage_common_050_20/metrics.json'),
        'production_coverage': str(workflow['PROD_COVER']),
        'candidate_impact': str(root / 'candidate_impact_common_050_20/pipeline_fp_impact.json'),
        'production_impact': str(workflow['PROD_IMPACT']),
        'candidate_gate': str(root / 'candidate_impact_common_050_20/validation_gate_result.json'),
        'production_gate': str(workflow['WORK'] / 'prod_current_impact_050_20/validation_gate_result.json'),
        'production_cache_verification': str(workflow['WORK'] / 'prod_full10_cache_verified.json'),
    },
}
workflow['write'](root / 'full10_common_policy_vs_prod.json', summary)
print(json.dumps({'status': 'complete', 'result': str(root / 'full10_common_policy_vs_prod.json'),
                  'matches': len(match_rows), 'candidate_gate': reports['candidate_contract']['status']}, indent=2))
