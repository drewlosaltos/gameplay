"""Dataset paths resolve under SKYBALL_DATA_ROOT."""
import json
from pathlib import Path

from skyball_gameplay.data_root import DEFAULT_DATA_ROOT, data_root, dataset_path

REPO = Path(__file__).parents[1]


def test_relative_paths_resolve_under_the_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv('SKYBALL_DATA_ROOT', raising=False)
    assert data_root() == Path(DEFAULT_DATA_ROOT)
    monkeypatch.setenv('SKYBALL_DATA_ROOT', str(tmp_path))
    assert dataset_path('derived/gameplay/v') == tmp_path / 'derived/gameplay/v'
    assert dataset_path('/elsewhere/v') == Path('/elsewhere/v')


def test_canonical_config_and_contract_name_the_same_relative_dataset():
    recipe = json.loads((REPO / 'config/indoor_canonical.json').read_text())
    contract = json.loads((REPO / recipe['split_contract']).read_text())
    assert recipe['dataset'] == contract['dataset_root']
    assert recipe['dataset'].startswith('derived/gameplay/')


def test_qwen_snapshot_override_and_hub_rerooting(monkeypatch, tmp_path):
    from skyball_gameplay.data_root import hf_snapshot_path, qwen_snapshot

    stored = '/home/someone/.cache/huggingface/hub/models--Qwen--X/snapshots/abc'
    monkeypatch.delenv('SKYBALL_QWEN_SNAPSHOT', raising=False)
    monkeypatch.delenv('HF_HUB_CACHE', raising=False)
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    assert qwen_snapshot(stored) == tmp_path / 'hub/models--Qwen--X/snapshots/abc'
    assert hf_snapshot_path('Qwen/X', 'abc') == tmp_path / 'hub/models--Qwen--X/snapshots/abc'
    existing = tmp_path / 'local'
    existing.mkdir()
    assert qwen_snapshot(existing) == existing
    monkeypatch.setenv('SKYBALL_QWEN_SNAPSHOT', '/override')
    assert qwen_snapshot(stored) == Path('/override')
