#!/usr/bin/env python3
"""Canonical full-frame legacy adaptation, human fine-tuning, and two-source validation.

Run in pt5090new inside tmux, after dataset review/approval. Outputs must be new.
--dry-run prints the complete plan without starting jobs or writing outputs.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from skyball_gameplay.full10 import validate_full10_inputs


def training_args(dataset, initializer, output, epochs, step, port, microbatch=64):
    return ['--timm-model', 'vit_base_patch16_siglip_512.v2_webli',
        '--optimizer', 'adamw', '--lr', '2e-5', '--backbone-lr-multiplier', '.05',
        '--weight-decay', '1e-4', '--epochs', str(epochs), '--lr-scheduler', 'step',
        '--lr-step-size', str(step), '--lr-gamma', '.1', '--amp', '--validation-fp32',
        '--dist-url', f'tcp://127.0.0.1:{port}', '--dist-backend', 'nccl',
        '--multiprocessing-distributed', '--world-size', '1', '--rank', '0',
        '-j', '16', '-b', str(microbatch), '--grad-accum-steps', str(128//microbatch), '--input-frames', '4',
        '--label-mode', 'binary', '--image-size', '512', '512', '--train-geometry',
        'full_frame', '--aug-policy', 'none', '--random-erasing', '0',
        '--color-jitter', '.1', '--temporal-dropout', '.1', '--temporal-shared-augment',
        '--selection-metric', 'f1', '--seed', '20260916', '--init-checkpoint',
        str(initializer), '--output-dir', str(output), '--run-tag', 'fullframe', str(dataset)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy-data', type=Path, required=True)
    p.add_argument('--human-data', type=Path, required=True)
    p.add_argument('--init-checkpoint', type=Path, required=True)
    p.add_argument('--production-suite', type=Path, required=True,
                   help='Completed canonical full10 production run with saved gameplay and GT provenance')
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--gate-threshold', type=float, default=.15)
    p.add_argument('--gate-padding-sec', type=float, default=10.)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    for key in ['legacy_data', 'human_data', 'init_checkpoint', 'production_suite', 'output_root']:
        setattr(args, key, getattr(args, key).resolve())
    if args.output_root.exists():
        p.error('Use a new versioned output directory')
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or len(gpus) not in (1,2,4):
        p.error('Use 1, 2 or 4 unique GPUs; accumulation preserves batch 128')
    for dataset in [args.legacy_data, args.human_data]:
        if not (dataset/'metadata/windows.jsonl').is_file():
            p.error(f'Missing dataset manifest: {dataset}')
    if not args.init_checkpoint.is_file():
        p.error('Missing initializer checkpoint')
    validate_full10_inputs(args.production_suite)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    legacy = args.output_root/'legacy_adaptation'
    human = args.output_root/'human_finetune'
    stages = [training_args(args.legacy_data, args.init_checkpoint, legacy, 2, 1, port, 16*len(gpus)),
              training_args(args.human_data, legacy/'best_fullframe.pth.tar', human, 4, 2, port, 16*len(gpus))]
    plan = {'recipe': 'siglip512_two_stage_fullframe_v1',
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            'stages': stages, 'required_validation_sources': ['classification', 'canonical_full10'],
            'classification_threshold': .5, 'effective_batch': 128,
            'checkpoint_selection': 'F1; stage one on legacy val, stage two on human val',
            'note': 'Dataset approval and source-video split integrity are prerequisites. No new annotation review or production promotion is implied by this launcher.'}
    if args.dry_run:
        print(json.dumps(plan, indent=2)); return
    args.output_root.mkdir(parents=True)
    for label, path in [('initializer', args.init_checkpoint),
                        ('legacy_manifest', args.legacy_data/'metadata/windows.jsonl'),
                        ('human_manifest', args.human_data/'metadata/windows.jsonl')]:
        plan[label+'_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output_root/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    snapshot = args.output_root/'code_snapshot'; snapshot.mkdir()
    source_files = [*REPO.glob('skyball_gameplay/*.py'), Path(__file__),
                    REPO/'scripts/score_gameplay_pipeline_inputs.py',
                    REPO/'scripts/evaluate_gameplay_event_coverage.py',
                    REPO/'scripts/evaluate_gameplay_gate.py',
                    REPO/'validation/gameplay_event_coverage_full10_v1.json']
    for path in source_files:
        dest = snapshot/path.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    env = os.environ | {'CUDA_VISIBLE_DEVICES': args.gpus, 'OMP_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1'}
    subprocess.run(['nvidia-smi'], check=True)
    def run(command, log_path, job_env=env):
        with log_path.open('w') as log:
            subprocess.run([sys.executable, *command], cwd=REPO, env=job_env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    for index, stage in enumerate(stages):
        print('TRAIN_STAGE', index + 1, flush=True)
        run(['-m', 'skyball_gameplay.training', *stage], args.output_root/f'stage{index+1}.log')
    checkpoint = human/'best_fullframe.pth.tar'
    evaluations = {}
    for model, ckpt in [('initializer', args.init_checkpoint), ('candidate', checkpoint)]:
        for population, data in [('legacy', args.legacy_data), ('human', args.human_data)]:
            dest = args.output_root/f'{model}_{population}_validation'
            run(['-m', 'skyball_gameplay.evaluation', '--data', str(data), '--checkpoint',
                 str(ckpt), '--output-dir', str(dest), '--save-predictions', '-b', '32', '-j', '8'],
                args.output_root/f'{model}_{population}_eval.log', env | {'CUDA_VISIBLE_DEVICES':gpus[0]})
            evaluations[f'{model}_{population}'] = str(dest/'metrics.json')
    jobs = []
    for shard, gpu in enumerate(gpus):
        log = (args.output_root/f'full10_shard{shard}.log').open('w')
        process = subprocess.Popen([sys.executable, 'scripts/score_gameplay_pipeline_inputs.py',
            '--checkpoint', str(checkpoint), '--pipeline-root', str(args.production_suite),
            '--output-root', str(args.output_root/'full10_scores'), '--shard', str(shard),
            '--num-shards', str(len(gpus))], cwd=REPO,
            env=env | {'CUDA_VISIBLE_DEVICES':gpu}, stdout=log, stderr=subprocess.STDOUT)
        jobs.append((process,log))
    codes=[]
    for process,log in jobs: codes.append(process.wait());log.close()
    if any(codes): raise RuntimeError(f'Full10 scoring failed: {codes}')
    run(['scripts/evaluate_gameplay_event_coverage.py', '--pipeline-root', str(args.production_suite),
        '--candidate-scores', str(args.output_root/'full10_scores'), '--output-root', str(args.output_root/'full10_validation'),
        '--candidate-threshold', str(args.gate_threshold), '--candidate-padding-sec', str(args.gate_padding_sec)],
        args.output_root/'full10_validation.log')
    ledger = {'status':'complete', 'checkpoint':str(checkpoint),
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'classification':evaluations, 'full10':str(args.output_root/'full10_validation/metrics.json'),
              'production_promoted':False}
    (args.output_root/'result.json').write_text(json.dumps(ledger,indent=2)+'\n')
    print(json.dumps(ledger,indent=2))


if __name__ == '__main__':
    main()
