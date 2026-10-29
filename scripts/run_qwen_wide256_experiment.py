"""Run the versioned unfrozen aspect-preserving Qwen gameplay pilot and both required validation sources.

Copy this controller and the gameplay source into a run directory before launch.
Never edit a launched controller. Configuration is recorded in config.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    root = Path(sys.argv[1]).resolve()
    config = json.loads((root / 'config.json').read_text())
    code = root / 'code_snapshot'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=config['gpus'],
               PYTHONPATH=f"{config['runtime_overlay']}:{code}",
               OMP_NUM_THREADS='4', NVIDIA_TF32_OVERRIDE='0', PYTHONUNBUFFERED='1')
    env['PATH'] = str(Path(config['python']).parent) + ':' + env['PATH']
    python = config['python']
    started = time.time()
    completed = False
    commands = []

    def status(phase, **extra):
        write_json(root / 'control_status.json', dict(phase=phase, updated_at=time.time(),
                   started_at=started, gpus=config['gpus'], **extra))

    def run(phase, command, gpu=None):
        status(phase)
        entry = dict(phase=phase, command=command, gpus=gpu or config['gpus'], started_at=time.time())
        commands.append(entry)
        write_json(root / 'commands.json', commands)
        with (root / f'{phase}.log').open('x') as handle:
            result = subprocess.run(command, cwd=code, env=dict(env, CUDA_VISIBLE_DEVICES=entry['gpus']),
                                    stdout=handle, stderr=subprocess.STDOUT)
        entry.update(returncode=result.returncode, elapsed_sec=time.time()-entry['started_at'])
        write_json(root / 'commands.json', commands)
        if result.returncode:
            raise RuntimeError(f'{phase} failed with exit code {result.returncode}; see {phase}.log')

    try:
        status('verifying_inputs')
        preflight = json.loads((root / 'preflight.json').read_text())
        assert preflight['passed']
        for dataset in preflight['datasets'].values():
            assert digest(dataset['manifest']) == dataset['manifest_sha256']
        for relative, sha in config['code_sha256'].items():
            assert digest(code / relative) == sha, relative
        assert digest(config['initial_checkpoint']) == config['initial_checkpoint_sha256']
        assert digest(config['baseline_checkpoint']) == config['baseline_checkpoint_sha256']
        assert digest(config['baseline_record']) == config['baseline_record_sha256']
        assert digest(config['baseline_full10']) == config['baseline_full10_sha256']
        common = [python, '-m', 'skyball_gameplay.training', '--arch', 'qwen_vision',
                  '--qwen-snapshot', config['qwen_snapshot'],
                  '--optimizer', 'adamw', '--lr', '1e-4', '--backbone-lr-multiplier', '.1', '--weight-decay', '1e-4',
                  '--warmup-epochs', '.5', '--max-grad-norm', '1',
                  '--amp', '--amp-dtype', 'bfloat16', '--validation-fp32',
                  '--dist-url', config['dist_url'], '--dist-backend', 'nccl',
                  '--multiprocessing-distributed', '--world-size', '1', '--rank', '0',
                  '-j', '12', '-b', '16', '--grad-accum-steps', '8',
                  '--input-frames', '4', '--label-mode', 'binary', '--image-size', '192', '352', '--resize-mode', 'letterbox',
                  '--train-geometry', 'full_frame', '--aug-policy', 'none', '--random-erasing', '0',
                  '--color-jitter', '.1', '--temporal-dropout', '.1', '--temporal-shared-augment',
                  '--selection-metric', 'f1', '--seed', '20260916', '--print-freq', '100',
                  '--lr-scheduler', 'step', '--lr-gamma', '.1', '--run-tag', 'qwen_unfrozen_wide']
        legacy = root / 'legacy_adaptation'
        human = root / 'human_finetune'
        legacy.mkdir(exist_ok=False)
        human.mkdir(exist_ok=False)
        run('train_legacy', common + [config['legacy_data'], '--epochs', '2', '--lr-step-size', '1',
                                     '--output-dir', str(legacy), '--init-checkpoint', config['initial_checkpoint']])
        run('train_human', common + [config['human_data'], '--epochs', '4', '--lr-step-size', '2',
                                    '--init-checkpoint', str(legacy / 'best_qwen_unfrozen_wide.pth.tar'),
                                    '--output-dir', str(human)])
        checkpoint = human / 'best_qwen_unfrozen_wide.pth.tar'
        gpus = config['gpus'].split(',')
        for name in ['human', 'legacy']:
            run(f'evaluate_{name}', [python, '-m', 'skyball_gameplay.evaluation',
                '--data', config[f'{name}_data'], '--checkpoint', str(checkpoint),
                '--output-dir', str(root / f'validation_{name}'), '-b', '8', '-j', '6',
                '--save-predictions'], gpus[0])
        status('score_full10')
        jobs = []
        try:
            for shard, gpu in enumerate(gpus):
                command = [python, str(code / 'scripts/score_gameplay_pipeline_inputs.py'),
                           '--shard', str(shard), '--num-shards', str(len(gpus)),
                           '--checkpoint', str(checkpoint), '--pipeline-root', config['pipeline_root'],
                           '--output-root', str(root / 'full10_scores')]
                entry = dict(phase=f'full10_shard{shard}', command=command, gpus=gpu, started_at=time.time())
                commands.append(entry)
                handle = (root / f'full10_shard{shard}.log').open('x')
                process = subprocess.Popen(command, cwd=code, env=dict(env, CUDA_VISIBLE_DEVICES=gpu),
                                           stdout=handle, stderr=subprocess.STDOUT)
                jobs.append((process, handle, entry))
            write_json(root / 'commands.json', commands)
            for process, handle, entry in jobs:
                entry.update(returncode=process.wait(), elapsed_sec=time.time()-entry['started_at'])
                handle.close()
            write_json(root / 'commands.json', commands)
            if any(entry['returncode'] for _, _, entry in jobs):
                raise RuntimeError('Full10 scoring failed; see shard logs')
        finally:
            for process, handle, _ in jobs:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                handle.close()
        run('evaluate_full10', [python, str(code / 'scripts/evaluate_gameplay_event_coverage.py'),
            '--pipeline-root', config['pipeline_root'], '--candidate-scores', str(root / 'full10_scores'),
            '--output-root', str(root / 'full10_validation'), '--candidate-threshold', '.15',
            '--candidate-padding-sec', '10'])
        sys.path.insert(0, str(code))
        from skyball_gameplay.evaluation import binary_metrics
        baseline = json.loads(Path(config['baseline_record']).read_text())
        baseline_gate = json.loads(Path(config['baseline_full10']).read_text())
        gate = json.loads((root / 'full10_validation/metrics.json').read_text())
        for old, new in zip(sorted(baseline_gate['per_match'], key=lambda x:x['match']),
                            sorted(gate['per_match'], key=lambda x:x['match']), strict=True):
            assert old['match'] == new['match']
            for field in ['events', 'rallies', 'frames', 'fps']:
                assert old['models']['candidate_selected_policy'][field] == new['models']['candidate_selected_policy'][field]
        predictions = []
        classification = {}
        for name, expected in [('human', 1420), ('legacy', 15800)]:
            metrics = json.loads((root / f'validation_{name}/metrics.json').read_text())
            assert metrics['binary']['count'] == expected
            classification[name] = metrics['binary']
            with (root / f'validation_{name}/predictions.csv').open() as handle:
                predictions.extend(csv.DictReader(handle))
        classification['combined'] = binary_metrics(
            [int(row['binary_target']) for row in predictions],
            [int(row['binary_prediction']) for row in predictions],
            [float(row['score_gameplay']) for row in predictions])
        baseline_classification = dict(human=baseline['human_validation'],
            legacy=baseline['legacy_validation']['candidate'], combined=baseline['combined_validation']['candidate'])
        comparison = {name: dict(candidate_f1=classification[name]['f1'],
            baseline_f1=baseline_classification[name]['f1'],
            delta_f1=classification[name]['f1']-baseline_classification[name]['f1'],
            meets_or_exceeds_baseline=classification[name]['f1']>=baseline_classification[name]['f1'])
            for name in classification}
        result = dict(status='complete', model_id=config['run_id'], checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint), classification=classification,
            baseline_classification=baseline_classification, classification_comparison=comparison,
            full10=dict(candidate=gate['aggregate']['candidate_selected_policy'],
                        baseline=baseline_gate['aggregate']['candidate_selected_policy'],
                        policy=gate['policies']['candidate_selected_policy'],
                        report=str(root / 'full10_validation/metrics.json')),
            selected_epochs={name: [json.loads(line) for line in (directory / 'epoch_metrics.jsonl').read_text().splitlines()]
                             for name, directory in [('legacy', legacy), ('human', human)]},
            elapsed_sec=time.time()-started, commands=str(root/'commands.json'), config=str(root/'config.json'),
            promotion='Not promoted; this pilot does not include the target-FP safety/effectiveness gate required for a production recommendation.')
        write_json(root / 'result.json', result)
        evidence = Path(config['repo']) / 'validation' / f"{config['run_id']}.json"
        if evidence.exists():
            raise FileExistsError(evidence)
        write_json(evidence, result)
        rows = ['# Frozen unfrozen aspect-preserving Qwen gameplay pilot', '', '| Validation | Qwen F1 | SigLIP F1 | Delta, pp |',
                '|---|---:|---:|---:|']
        for name, metrics in comparison.items():
            rows.append(f"| {name} | {metrics['candidate_f1']:.4%} | {metrics['baseline_f1']:.4%} | {100*metrics['delta_f1']:+.4f} |")
        rows += ['', 'Full10 uses threshold 0.15, padding 10 seconds per side, merge gap 0 seconds.', '']
        for name in ['candidate', 'baseline']:
            values = result['full10'][name]
            rows.append(f"- {name}: {values['events_excluded']}/{values['events']} confirmed events excluded; "
                        f"{values['outside_event_margin_admitted_pct_match']:.4f}% extra footage / match duration.")
        rows += ['', f"Elapsed run time: {result['elapsed_sec']:.1f} seconds. Per-match inference times: full10_scores/*/metadata.json.",
                 '', result['promotion'], '', f'Full evidence: {root / "result.json"}', '']
        (root / 'REPORT.md').write_text('\n'.join(rows))
        completed = True
        status('completed', elapsed_sec=time.time()-started, result=str(root/'result.json'))
    except BaseException:
        status('failed', error=traceback.format_exc())
        raise
    finally:
        (root / 'exit_code').write_text('0\n' if completed else '1\n')


if __name__ == '__main__':
    main()
