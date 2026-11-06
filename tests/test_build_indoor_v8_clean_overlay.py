"""The clean overlay drops each confirmed re-encode from its own side only."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/build_indoor_v8_clean_overlay.py'


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))


def fixture(tmp_path):
    base = tmp_path / 'base/metadata'
    base.mkdir(parents=True)
    videos = [('train_a', 'train'), ('dup', 'train'), ('twin', 'val'), ('old_val', 'val'), ('new_val', 'val')]
    windows = [{'video_id': vid, 'split': split, 'class_name': 'gameplay'} for vid, split in videos]
    write_jsonl(base / 'windows.jsonl', windows)
    write_jsonl(base / 'images.jsonl', [{'source_video': f'{row["video_id"]}.mp4', 'split': row['split']}
                                        for row in windows for _ in range(4)])
    (base / 'summary.json').write_text('{"version": "base"}')
    (base / 'class_to_idx.json').write_text('{"gameplay": 0}')
    split = tmp_path / 'split.json'
    split.write_text(json.dumps({'train': ['dup', 'train_a'], 'val20': ['old_val', 'twin'], 'new30': ['new_val'],
                                 'val50': ['new_val', 'old_val', 'twin'], 'counts': {}}))
    audit = tmp_path / 'leak.json'
    audit.write_text(json.dumps({'status': 'confirmed_validation_leak',
                                 'confirmed_reencoded_pairs': [{'training_video': 'dup', 'validation_video': 'twin'}]}))
    return ['--base-metadata', str(base), '--base-dataset', 'base', '--split', str(split), '--leak-audit', str(audit)]


def run(args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], check=True, text=True, capture_output=True)


def test_overlay_removes_confirmed_reencodes(tmp_path):
    output = tmp_path / 'clean'
    run(fixture(tmp_path) + ['--output', str(output)])
    kept = [json.loads(line)['video_id'] for line in (output / 'metadata/windows.jsonl').read_text().splitlines()]
    assert kept == ['train_a', 'old_val', 'new_val']
    assert len((output / 'metadata/images.jsonl').read_text().splitlines()) == 12
    split = json.loads((output / 'metadata/split.json').read_text())
    assert split['val50'] == ['new_val', 'old_val'] and split['train'] == ['train_a']
    summary = json.loads((output / 'metadata/summary.json').read_text())
    assert summary['removed_windows'] == {'dup': 1, 'twin': 1}
    assert summary['source_video_counts'] == {'train': 1, 'val20': 1, 'new30': 1}
    assert not (output / 'train').exists()


def test_overlay_refuses_an_existing_root(tmp_path):
    output = tmp_path / 'clean'
    output.mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        run(fixture(tmp_path) + ['--output', str(output)])
