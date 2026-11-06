"""Guard the rally recipe launcher and per-epoch checkpoint retention."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from skyball_gameplay.training import save_checkpoint

REPO = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('rally', REPO / 'scripts/train_rally_from_prod.py')
rally = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rally)


def recipes():
    recipe = json.loads((REPO / 'config/rally_canonical.json').read_text())
    return recipe, json.loads((REPO / recipe['base_recipe']).read_text())


def command(gpus):
    args = argparse.Namespace(data=Path('/data/rally'), output_dir=Path('/out/run'), gpus=gpus,
                              init_checkpoint=Path('/ckpt/prod.pth.tar'), epochs='4',
                              dist_url='tcp://127.0.0.1:1', extra=[])
    return rally.build_command(args, *recipes())


def value(cmd, flag):
    return cmd[len(cmd) - 1 - cmd[::-1].index(flag) + 1]


@pytest.mark.parametrize('gpus,batch,accum', [('0,1', '8', '2'), ('0,1,2,3', '16', '1')])
def test_effective_batch_stays_16(gpus, batch, accum):
    cmd = command(gpus)
    assert value(cmd, '-b') == batch and value(cmd, '--grad-accum-steps') == accum


def test_rally_overrides_on_canonical_recipe():
    cmd = command('0,1')
    assert cmd[cmd.index('skyball_gameplay.training') + 1] == '/data/rally'
    assert value(cmd, '--init-checkpoint') == '/ckpt/prod.pth.tar'
    assert value(cmd, '--output-dir') == '/out/run'
    assert value(cmd, '--run-tag') == 'rally_qwen672x384_from_prod_v8'
    assert '--keep-epoch-checkpoints' in cmd
    assert value(cmd, '--image-size') == '384' and value(cmd, '--resize-mode') == 'letterbox'


def test_rejects_gpu_count_that_breaks_effective_batch():
    with pytest.raises(ValueError, match='effective batch'):
        command('0,1,2')


def test_epoch_checkpoints_are_independent_copies(tmp_path):
    save_checkpoint({'epoch': 1}, True, tag='t', output_dir=tmp_path, epoch=1)
    save_checkpoint({'epoch': 2}, False, tag='t', output_dir=tmp_path, epoch=2)
    assert torch.load(tmp_path / 'epoch1_t.pth.tar')['epoch'] == 1
    assert torch.load(tmp_path / 'epoch2_t.pth.tar')['epoch'] == 2
    assert torch.load(tmp_path / 'best_t.pth.tar')['epoch'] == 1
    assert torch.load(tmp_path / 'last_t.pth.tar')['epoch'] == 2


def test_no_epoch_copies_by_default(tmp_path):
    save_checkpoint({'epoch': 1}, True, tag='t', output_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['best_t.pth.tar', 'last_t.pth.tar']
