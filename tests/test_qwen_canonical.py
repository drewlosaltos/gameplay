"""Guard the canonical split and checkpoint-compatible FP8 conversion."""
import importlib.util
import json
from pathlib import Path
import pytest
import torch

spec = importlib.util.spec_from_file_location('canonical', Path(__file__).parents[1] / 'scripts/train_qwen_canonical.py')
canonical = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canonical)


def fixture_split(tmp_path, train='training', val='validation'):
    (tmp_path / 'metadata').mkdir()
    records = [{'source_video': f'{source}.mp4', 'split': split, 'frame_paths': ['frame.jpg'] * 4}
               for source, split in [(train, 'train'), (val, 'val')]]
    (tmp_path / 'metadata/windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    return {'clean_val48': ['validation'], 'excluded_train_reencodes': ['duplicate'],
            'excluded_val20_reencodes': ['contaminated'], 'dataset_root': '/unused'}


def test_manifest_validation_rejects_overlap(tmp_path):
    contract = fixture_split(tmp_path, train='validation')
    with pytest.raises(ValueError, match='overlap'):
        canonical.validate_split(tmp_path, contract)


def test_manifest_validation_rejects_reencoded_leak(tmp_path):
    contract = fixture_split(tmp_path, train='duplicate')
    with pytest.raises(ValueError, match='reencode'):
        canonical.validate_split(tmp_path, contract)


def test_manifest_validation_rejects_changed_membership(tmp_path):
    contract = fixture_split(tmp_path, val='unexpected')
    with pytest.raises(ValueError, match='membership'):
        canonical.validate_split(tmp_path, contract)


def test_manifest_audit_ignores_extra_images(tmp_path):
    contract = fixture_split(tmp_path)
    (tmp_path / 'train').mkdir()
    (tmp_path / 'train/excluded.jpg').write_bytes(b'excluded')
    result = canonical.validate_split(tmp_path, contract)
    assert result['window_counts'] == {'train': 1, 'val': 1}


def test_fp8_conversion_preserves_checkpoint_and_fp32_eval():
    pytest.importorskip('torchao.float8')
    from skyball_gameplay.acceleration import prepare_training_acceleration
    model = torch.nn.Module()
    model.encoder = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 32))
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    inputs = torch.randn(2, 32)
    model.eval()
    expected = model.encoder(inputs)
    result = prepare_training_acceleration(model, 'compile_fp8')
    assert result['fp8_linear_count'] == 2
    assert list(model.state_dict()) == list(weights)
    torch.testing.assert_close(model.encoder(inputs), expected, rtol=0, atol=0)
    plain = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU(), torch.nn.Linear(32, 32))
    plain.load_state_dict(model.encoder.state_dict(), strict=True)
    torch.testing.assert_close(plain(inputs), expected, rtol=0, atol=0)
