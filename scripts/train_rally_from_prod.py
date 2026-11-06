#!/usr/bin/env python3
"""Train the rally classifier: canonical Qwen v8 recipe, initialized from prod gameplay.

Same architecture/augmentation/optimizer as ``config/indoor_canonical.json``, with
three changes: the rally dataset, ``--init-checkpoint`` = production v8 gameplay
checkpoint, and two GPUs at per-GPU batch 4 x grad-accum 2 (effective batch 16,
equal to the canonical recipe).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
PROD = Path('/mnt/t/output/trn/skyball_gameplay/v8_val48_trainclean_qwen672x384_single_stage_4gpu_20261006_r3/'
            'best_qwen672x384_v8_single_stage_4gpu.pth.tar')


def set_arg(options: list[str], flag: str, value: str) -> None:
    if flag in options:
        options[options.index(flag) + 1] = value
    else:
        options += [flag, value]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', default='2,3')
    p.add_argument('--init-checkpoint', type=Path, default=PROD)
    p.add_argument('--epochs', default='4')
    p.add_argument('--dist-url', default='tcp://127.0.0.1:50291')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Use a new versioned output directory')
    recipe = json.loads((REPO / 'config/indoor_canonical.json').read_text())
    options = list(recipe['args'])
    gpus = args.gpus.split(',')
    set_arg(options, '--output-dir', str(args.output_dir))
    set_arg(options, '--dist-url', args.dist_url)
    set_arg(options, '-b', str(4 * len(gpus)))
    set_arg(options, '--grad-accum-steps', str(16 // (4 * len(gpus))))
    set_arg(options, '--epochs', args.epochs)
    set_arg(options, '--run-tag', 'rally_qwen672x384_from_prod_v8')
    options += ['--init-checkpoint', str(args.init_checkpoint), *args.extra]
    command = [sys.executable, '-m', 'skyball_gameplay.training', str(args.data), *options]
    args.output_dir.mkdir(parents=True)
    plan = {'recipe': recipe['recipe_id'] + '+rally_from_prod_v8', 'dataset': str(args.data),
            'init_checkpoint': str(args.init_checkpoint), 'gpus': gpus, 'command': command,
            'gpu_before_launch': subprocess.check_output(['nvidia-smi'], text=True)}
    (args.output_dir / 'rally_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    with (args.output_dir / 'launcher.log').open('x') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
