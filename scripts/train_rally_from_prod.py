#!/usr/bin/env python3
"""Train the rally classifier (``config/rally_canonical.json``).

The canonical gameplay recipe (``config/indoor_canonical.json``) with these changes:
the rally dataset, ``--init-checkpoint`` = production v8 gameplay checkpoint,
per-GPU batch 4 with gradient accumulation to an effective batch of 16, and a
checkpoint kept for every epoch so epochs can be selected on full10.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / 'config/rally_canonical.json'
sys.path.insert(0, str(REPO))
from skyball_gameplay.data_root import dataset_path  # noqa: E402


def set_arg(options: list[str], flag: str, value: str) -> None:
    if flag in options:
        options[options.index(flag) + 1] = value
    else:
        options += [flag, value]


def build_command(args: argparse.Namespace, recipe: dict, base: dict) -> list[str]:
    gpus = args.gpus.split(',')
    per_gpu, effective = recipe['per_gpu_batch'], recipe['effective_batch']
    if effective % (per_gpu * len(gpus)):
        raise ValueError(f'{len(gpus)} GPUs x batch {per_gpu} does not divide effective batch {effective}')
    options = list(base['args'])
    set_arg(options, '--output-dir', str(args.output_dir))
    set_arg(options, '--dist-url', args.dist_url)
    set_arg(options, '-b', str(per_gpu * len(gpus)))
    set_arg(options, '--grad-accum-steps', str(effective // (per_gpu * len(gpus))))
    set_arg(options, '--epochs', args.epochs)
    set_arg(options, '--run-tag', recipe['run_tag'])
    options += ['--init-checkpoint', str(args.init_checkpoint), *recipe['extra_args'], *args.extra]
    return [sys.executable, '-m', 'skyball_gameplay.training', str(args.data), *options]


def main() -> None:
    recipe = json.loads(RECIPE.read_text())
    base = json.loads((REPO / recipe['base_recipe']).read_text())
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=dataset_path(recipe['dataset']))
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpus', default='0,1')
    p.add_argument('--init-checkpoint', type=Path, default=Path(recipe['init_checkpoint']))
    p.add_argument('--epochs', default='4')
    p.add_argument('--dist-url', default='tcp://127.0.0.1:50291')
    p.add_argument('--dry-run', action='store_true', help='print the command and exit')
    p.add_argument('--extra', nargs=argparse.REMAINDER, default=[], help='extra training args, appended last')
    args = p.parse_args()
    if args.output_dir.exists():
        p.error('Use a new versioned output directory')
    if not (args.data / 'metadata/windows.jsonl').is_file():
        p.error(f'No rally dataset at {args.data}')
    command = build_command(args, recipe, base)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    gpus = args.gpus.split(',')
    args.output_dir.mkdir(parents=True)
    plan = {'recipe': f"{recipe['recipe_id']} (base {base['recipe_id']})", 'dataset': str(args.data),
            'init_checkpoint': str(args.init_checkpoint), 'gpus': gpus, 'command': command,
            'gpu_before_launch': subprocess.check_output(['nvidia-smi'], text=True)}
    (args.output_dir / 'rally_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    with (args.output_dir / 'launcher.log').open('x') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
