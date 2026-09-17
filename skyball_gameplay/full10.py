"""Frozen canonical full-video validation membership and input bindings."""
import hashlib
import json
from pathlib import Path


CONTRACT = Path(__file__).resolve().parents[1] / "validation/gameplay_event_coverage_full10_v1.json"


def validate_full10_inputs(root):
    contract = json.loads(CONTRACT.read_text())
    expected = {row['match']: row for row in contract['matches']}
    paths = sorted(Path(root).glob('eval_*/manifest.json'))
    if {path.parent.name for path in paths} != set(expected):
        raise ValueError('Canonical full10 match membership differs; create a new versioned contract')
    for path in paths:
        actual = json.loads(path.read_text())
        row = expected[path.parent.name]
        if actual.get('run_status') != 'succeeded':
            raise ValueError(f'Incomplete baseline: {path}')
        for key in ['video', 'gt_sync_csv']:
            if Path(actual[key]).resolve() != Path(row[key]).resolve():
                raise ValueError(f'Canonical {key} binding differs: {path}')
        gt = Path(actual['gt_sync_csv'])
        digest = hashlib.sha256(gt.read_bytes()).hexdigest()
        if digest != row['gt_sha256'] or digest != actual['inputs']['gt_sync_csv']['sha256']:
            raise ValueError(f'Canonical GT changed: {gt}')
        if Path(actual['video']).stat().st_size != row['video_size_bytes']:
            raise ValueError(f'Canonical video size differs: {path}')
    return paths
