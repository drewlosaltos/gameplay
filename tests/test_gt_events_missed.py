from skyball_gameplay.training import gt_events_missed_metrics


def test_event_missed_only_when_every_covering_window_is_below_threshold():
    scores = [0.9, 0.2, 0.4, -1.0]
    covered = [["a:1", "b:2"], ["b:2", "c:3"], ["c:3"], ["d:4"]]
    m = gt_events_missed_metrics(scores, covered, thresholds=(0.5, 0.3))
    assert m["gt_events"] == 3          # d:4 only on an unscored window -> excluded
    assert m["gt_events_missed"] == 1   # c:3 max 0.4 < 0.5; b:2 admitted by 0.9 window
    assert m["gt_events_missed_ids"] == ["c:3"]
    assert m["gt_events_missed_by_threshold"] == {"0.5": 1, "0.3": 0}
    assert abs(m["gt_events_missed_pct"] - 100 / 3) < 1e-9


def test_no_events():
    m = gt_events_missed_metrics([0.1], [[]])
    assert m["gt_events"] == 0 and m["gt_events_missed"] == 0
