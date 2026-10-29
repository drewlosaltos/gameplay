"""Boundary and denominator checks for source-frame gameplay admission audits."""
import numpy as np
import pytest
from scripts.evaluate_gameplay_event_coverage import (
    aggregate, evaluate_prediction, frame_mask, ground_truth_masks, positive_ranges,
)


def event(frame, rally='1'):
    return {'synced_fnum': str(frame), 'set': '1', 'rally': rally}


def test_event_windows_union_clip_and_do_not_bridge_long_gaps():
    events = [event(2), event(4), event(18)]
    near, rally, _, _ = ground_truth_masks(events, 20, 1., 3.)
    assert np.flatnonzero(near).tolist() == list(range(7)) + list(range(14, 20))
    assert rally.all()


def test_event_exclusion_is_distinct_from_margin_exclusion():
    prediction = frame_mask([(5, 5)], 10)
    result = evaluate_prediction(prediction, [event(5)], 10, 1., 3.)
    assert result['events_excluded'] == 0
    assert result['event_margin']['frames']['fn'] == 6
    assert result['event_windows_fully_preserved'] == 0
    assert result['outside_event_margin_admitted_pct_match'] == 0


def test_extra_footage_denominator_is_entire_match():
    result = evaluate_prediction(np.ones(10, dtype=bool), [event(5)], 10, 1., 1.)
    assert result['outside_event_margin_admitted_pct_match'] == 70.
    assert result['event_margin']['outside_region_admission_rate'] == 1.
    shorter = evaluate_prediction(np.ones(10, dtype=bool), [event(5)], 10, 2., 1.)
    pooled = aggregate([result, shorter])
    assert pooled['outside_event_margin_admitted_pct_match'] == pytest.approx(100 * 9.5 / 15)


def test_production_threshold_padding_and_inclusive_frame_rounding():
    scores = [{'window_start_sec': '2', 'window_end_sec': '6', 'score_gameplay': '.2'},
              {'window_start_sec': '7', 'window_end_sec': '11', 'score_gameplay': '.199'}]
    ranges = positive_ranges(scores, 2.5, 30, .2, 1.)
    assert ranges == [(3, 18)]
    assert frame_mask(ranges, 30).sum() == 16


def test_merge_gap_is_applied_before_padding():
    scores = [{'window_start_sec': '0', 'window_end_sec': '4', 'score_gameplay': '1'},
              {'window_start_sec': '19', 'window_end_sec': '23', 'score_gameplay': '1'}]
    assert positive_ranges(scores, 1., 30, .5, 5., 10.) == [(1, 9), (15, 28)]
    assert positive_ranges(scores, 1., 30, .5, 5., 15.) == [(1, 28)]
