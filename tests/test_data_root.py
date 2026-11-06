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
