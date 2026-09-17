"""Reject a changed second-validation population before scoring/training."""
import hashlib
import json

import pytest

from skyball_gameplay import full10


def test_frozen_full10_rejects_changed_gt_and_membership(tmp_path, monkeypatch):
    video = tmp_path/'video.mp4'
    video.write_bytes(b'video')
    gt = tmp_path/'events.csv'
    gt.write_text('status,synced_fnum\nconfirmed,1\n')
    digest = hashlib.sha256(gt.read_bytes()).hexdigest()
    contract = tmp_path/'contract.json'
    contract.write_text(json.dumps({'matches': [{
        'match': 'eval_match', 'video': str(video), 'gt_sync_csv': str(gt),
        'gt_sha256': digest, 'video_size_bytes': 5}]}))
    monkeypatch.setattr(full10, 'CONTRACT', contract)
    folder = tmp_path/'suite/eval_match'
    folder.mkdir(parents=True)
    manifest = folder/'manifest.json'
    manifest.write_text(json.dumps({'run_status': 'succeeded', 'video': str(video),
        'gt_sync_csv': str(gt), 'inputs': {'gt_sync_csv': {'sha256': digest}}}))
    assert full10.validate_full10_inputs(folder.parent) == [manifest]
    gt.write_text('status,synced_fnum\nconfirmed,2\n')
    with pytest.raises(ValueError, match='GT changed'):
        full10.validate_full10_inputs(folder.parent)
    extra = folder.parent/'eval_extra'
    extra.mkdir()
    (extra/'manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='membership differs'):
        full10.validate_full10_inputs(folder.parent)
