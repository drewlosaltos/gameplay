#!/usr/bin/env python3
"""Launch the canonical single-stage Qwen FP8 recipe with clean val48."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
RECIPE = REPO / 'config/indoor_canonical.json'
sys.path.insert(0, str(REPO))
from skyball_gameplay.data_root import dataset_path, qwen_snapshot  # noqa: E402


def validate_split(data: Path, contract: dict) -> dict:
    """Audit manifest membership, known reencodes, and frozen V8 identity."""
    manifest = data / 'metadata/windows.jsonl'
    sources = {'train': set(), 'val': set()}
    counts = {'train': 0, 'val': 0}
    digest = hashlib.sha256()
    with manifest.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            split = 'train' if row['split'] == 'trn' else row['split']
            if split not in sources:
                raise ValueError(f'Unexpected split: {split}')
            sources[split].add(Path(row['source_video']).stem)
            counts[split] += 1
            if len(row['frame_paths']) != 4:
                raise ValueError('Canonical Qwen requires four-frame records')
    if sources['val'] != set(contract['clean_val48']):
        raise ValueError('Validation membership differs from clean val48')
    if sources['train'] & sources['val']:
        raise ValueError('Train/validation source overlap')
    if sources['train'] & set(contract['excluded_train_reencodes']):
        raise ValueError('Known validation reencode in training')
    if sources['val'] & set(contract['excluded_val20_reencodes']):
        raise ValueError('Contaminated legacy source in validation')
    if not counts['train']:
        raise ValueError('Empty training population')
    if data.resolve() == dataset_path(contract['dataset_root']).resolve():
        if digest.hexdigest() != contract['windows_sha256']:
            raise ValueError('Canonical V8 manifest identity changed')
    return {'windows_sha256': digest.hexdigest(), 'window_counts': counts,
            'source_counts': {k: len(v) for k, v in sources.items()}}


def build_command(args: argparse.Namespace, recipe: dict) -> list[str]:
    options = list(recipe['args'])
    replacements = {'--qwen-snapshot': str(args.qwen_snapshot),
                    '--output-dir': str(args.output_dir),
                    '--dist-url': args.dist_url}
    for flag, value in replacements.items():
        options[options.index(flag) + 1] = value
    return [sys.executable, '-m', 'skyball_gameplay.training', str(args.data), *options]


def launch(data: Path, snapshot: Path, output_dir: Path, gpus: list[str],
           dist_url: str = 'tcp://127.0.0.1:50235', dry_run: bool = False,
           record: dict | None = None) -> dict:
    """Audit inputs, then train; `record` is written as reproduction_provenance.json."""
    recipe = json.loads(RECIPE.read_text())
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(g.isdigit() for g in gpus):
        raise ValueError('The validated recipe requires four distinct GPU indices')
    if output_dir.exists():
        raise FileExistsError('Use a new versioned output directory')
    if not (snapshot / 'config.json').is_file():
        raise FileNotFoundError(f'Missing pretrained Qwen snapshot config under {snapshot}')
    contract = json.loads((REPO / recipe['split_contract']).read_text())
    audit = validate_split(data, contract)
    versions = {p: importlib.metadata.version(p) for p in ('torch', 'torchao', 'transformers')}
    args = argparse.Namespace(data=data, qwen_snapshot=snapshot, output_dir=output_dir, dist_url=dist_url)
    command = build_command(args, recipe)
    plan = {'recipe': recipe['recipe_id'], 'initialization': recipe['initialization'],
            'dataset': str(data), 'split_contract': recipe['split_contract'],
            'selection': 'clean val48 F1 only; full10 gate required after training',
            'split_audit': audit, 'gpus': gpus, 'versions': versions, 'command': command}
    if dry_run:
        return plan
    # Fail early for unavailable TorchAO/Qwen implementation and capture GPU state.
    from torchao.float8 import Float8LinearConfig
    Float8LinearConfig.from_recipe_name('tensorwise')
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel  # noqa: F401
    plan['gpu_before_launch'] = subprocess.check_output(['nvidia-smi'], text=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / 'canonical_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    # Freeze the package the run imports, as the 2026-10-07 production launcher did.
    shutil.copytree(REPO / 'skyball_gameplay', output_dir / 'code_snapshot/skyball_gameplay',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    if record is not None:
        (output_dir / 'reproduction_provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus), OMP_NUM_THREADS='4')
    env['PYTHONPATH'] = str(REPO) + os.pathsep + env.get('PYTHONPATH', '')
    with (output_dir / 'launcher.log').open('x') as log:
        subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return plan


def recipe_qwen_snapshot() -> Path:
    """The recipe's recorded snapshot, re-rooted to this machine (SKYBALL_QWEN_SNAPSHOT overrides)."""
    recipe = json.loads(RECIPE.read_text())
    return qwen_snapshot(recipe['args'][recipe['args'].index('--qwen-snapshot') + 1])


def main() -> None:
    recipe = json.loads(RECIPE.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=dataset_path(recipe['dataset']))
    parser.add_argument('--qwen-snapshot', type=Path, default=recipe_qwen_snapshot())
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:50235')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        plan = launch(args.data, args.qwen_snapshot, args.output_dir, args.gpus.split(','),
                      args.dist_url, args.dry_run)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    main()
