#!/usr/bin/env python3
"""Reproduce the selected indoor Qwen v5 four-epoch fine-tune."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA = Path('/mnt/t/data/vball/skyball/gameplay/human_app_v5_val20_20260925')
DEFAULT_INITIALIZER = Path(
    '/mnt/t/output/trn/skyball_gameplay/'
    'qwen36_unfrozen_wide352x192_two_stage_20260918_r1/'
    'legacy_adaptation/best_qwen_unfrozen_wide.pth.tar'
)
DEFAULT_SNAPSHOT = Path(
    '/home/atao/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/'
    'snapshots/95a723d08a9490559dae23d0cff1d9466213d989'
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--initializer', type=Path, default=DEFAULT_INITIALIZER)
    parser.add_argument('--qwen-snapshot', type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gpus', default='2,3')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 2 or len(set(gpus)) != 2:
        parser.error('This recorded recipe uses two distinct GPUs and effective batch 8')
    for path in (args.data / 'metadata/windows.jsonl', args.initializer,
                 args.qwen_snapshot / 'config.json'):
        if not path.is_file():
            parser.error(f'Missing training input: {path}')
    if args.output_dir.exists():
        parser.error(f'Use a new output directory: {args.output_dir}')
    command = [sys.executable, '-m', 'skyball_gameplay.training', str(args.data),
        '--arch', 'qwen_vision', '--qwen-snapshot', str(args.qwen_snapshot),
        '--optimizer', 'adamw', '--lr', '1e-4', '--backbone-lr-multiplier', '.01',
        '--weight-decay', '1e-4', '--epochs', '4', '--lr-scheduler', 'step',
        '--lr-step-size', '2', '--lr-gamma', '.1', '--amp', '--amp-dtype', 'bfloat16',
        '--validation-fp32', '--warmup-epochs', '.5', '--max-grad-norm', '1',
        '--dist-url', 'tcp://127.0.0.1:49855', '--dist-backend', 'nccl',
        '--multiprocessing-distributed', '--world-size', '1', '--rank', '0',
        '-j', '8', '-b', '4', '--grad-accum-steps', '2', '--input-frames', '4',
        '--label-mode', 'binary', '--image-size', '384', '672',
        '--resize-geometry', 'letterbox', '--rotation-degrees', '5',
        '--train-geometry', 'full_frame', '--color-jitter', '.1',
        '--temporal-dropout', '.1', '--temporal-shared-augment',
        '--aug-policy', 'none', '--random-erasing', '0',
        '--selection-metric', 'f1', '--seed', '20260916',
        '--init-checkpoint', str(args.initializer), '--run-tag', 'qwen672x384_v5',
        '--output-dir', str(args.output_dir)]
    plan = {
        'recipe': 'qwen36_672x384_v5_four_epoch_v1',
        'dataset': str(args.data), 'initializer': str(args.initializer),
        'qwen_snapshot': str(args.qwen_snapshot), 'gpus': gpus,
        'selection': 'best F1 on the frozen human val20 source-video split',
        'command': command,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'canonical_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus), OMP_NUM_THREADS='4')
    with (args.output_dir / 'launcher.log').open('x') as log:
        subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
