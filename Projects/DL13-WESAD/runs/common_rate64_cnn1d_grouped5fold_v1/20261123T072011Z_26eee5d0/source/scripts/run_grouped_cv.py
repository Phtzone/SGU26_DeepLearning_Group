"""CNN/GRU entry point: selected folds or full grouped CV; dry-run never trains."""
import argparse
import os
import re
import shutil
import psutil
import yaml
from src.data.common_rate import NOTEBOOK_METADATA, build_index, prepare_index
from src.data.preprocessing import prepare_dataset
from src.data.splits import load_manifest
from src.evaluation.aggregation import aggregate_group
from src.training.trainer import train_fold, completed_fold
from src.evaluation.enhanced_visualization import generate_visualizations
from src.utils.config import ROOT, raw_root, load_config, same_training_config, experiment_folder
from src.utils.console import ConsoleLogger
from src.utils.logging import RunLogger
from src.utils.system_info import system_info
from src.utils.tracking import identifier, write_json, read_json, digest, file_hash
from scripts.prepare_data import cache_path


def write_group_manifest(group, manifest):
    """Keep run lifecycle and an existing result copy in the same final state."""
    write_json(group / 'experiment_manifest.json', manifest)
    result = ROOT / 'results' / group.parent.name / group.name
    if result.is_dir():
        write_json(result / 'experiment_manifest.json', manifest)


def execution_plan(config, group_id=None, resume=False, folds=None):
    split = load_manifest(ROOT / config['split']['manifest'])
    if split['split_seed'] != 42:
        raise ValueError('Official experiments must use split_seed=42.')
    existing = None
    if resume:
        if not group_id or not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
            raise ValueError('--resume requires a valid --group-id.')
        existing = read_json(ROOT / 'runs' / experiment_folder(config) / group_id / 'experiment_manifest.json')
        if existing.get('runner_version') != 'tensorboard-manual-v1':
            raise ValueError('Historical/legacy groups are read-only. Start a new reproduction group.')
        if not same_training_config(existing['effective_config'], config) or existing['manifest_checksum'] != digest(split):
            raise ValueError('Resume effective config / split checksum mismatch.')
    selected = list(folds) if folds is not None else existing.get('requested_folds', []) if existing else [f['fold_id'] for f in split['folds']]
    available = {fold['fold_id'] for fold in split['folds']}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= available:
        raise ValueError(f'--folds must contain distinct IDs from {sorted(available)}.')
    selected = sorted(selected)
    subjects = split['original_subject_order']
    if config['data'].get('source') == 'common_rate':
        dataset = (ROOT / config['data']['processed_dir']).resolve()
        missing = [subject for subject in subjects
                   if not all((dataset / f'{subject}{suffix}').is_file() for suffix in ('_X.npy', '_quality_mask.npy'))]
        source_plan = {'data_source': str(dataset),
                       'missing_files': [name for name in NOTEBOOK_METADATA if not (dataset / name).is_file()]}
    else:
        dataset = raw_root()
        missing = [subject for subject in subjects if not (dataset / subject / (subject + '.pkl')).is_file()]
        source_plan = {'raw_root': str(dataset)}
    return split, selected, existing, {'group_id': group_id, 'resume': resume, 'selected_folds': selected,
        'protocol_folds': len(split['folds']), 'training_seeds': config['training']['seeds'],
        'effective_config': config, 'manifest_checksum': digest(split), **source_plan,
        'subjects_present': len(split['original_subject_order']) - len(missing), 'missing_subjects': missing,
        'tensorboard_log_root': str(ROOT / 'runs' / experiment_folder(config)),
        'subjects_by_fold': [fold for fold in split['folds'] if fold['fold_id'] in selected]}


def run(config, group_id=None, resume=False, folds=None, dry_run=False, experiment_path=None):
    if resume and not group_id:
        raise ValueError('--resume requires --group-id')
    group_id = group_id or identifier()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
        raise ValueError('Invalid group ID.')
    split, selected, existing, plan = execution_plan(config, group_id, resume, folds)
    group = ROOT / 'runs' / experiment_folder(config) / group_id
    result_directory = ROOT / 'results' / group.parent.name / group_id
    source = plan.get('data_source')
    exists = group.exists()
    if exists != resume:
        raise ValueError('New groups must not exist; resumed groups must already exist.')
    if dry_run:
        import json
        if source and not (plan['missing_subjects'] or plan['missing_files']):
            audit, _ = build_index(source, config['data'], split['original_subject_order'])
            plan['data_audit'] = audit
        print(json.dumps({'dry_run': True, 'training_executed': False, **plan}, indent=2))
        return plan
    if plan['missing_subjects'] or plan.get('missing_files'):
        raise FileNotFoundError(f"Dataset is incomplete at {source or plan['raw_root']}: "
                                f"subjects={plan['missing_subjects']}, metadata={plan.get('missing_files', [])}")
    if source and not resume and result_directory.exists():
        raise FileExistsError(f'Results already exist for this group: {result_directory}')
    visualizations_ready = not source or (result_directory / 'enhanced_visualizations/generation_manifest.json').is_file()
    if resume and existing['status'] in ('partial', 'completed') and (result_directory / 'summary.json').exists() and visualizations_ready:
        if all(completed_fold(group / f'seed_{seed}' / f'fold_{fold_id:02d}') for seed in config['training']['seeds'] for fold_id in selected):
            print('Selected folds already completed and verified; no artifacts were rewritten.')
            print('GROUP_ID=' + group_id)
            return group
    group.mkdir(parents=True, exist_ok=True)
    lock = group / '.lock'
    if lock.exists():
        previous = read_json(lock)
        try:
            alive = abs(psutil.Process(previous['pid']).create_time() - previous['created']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError('This experiment group is already active.')
        lock.unlink()
    with lock.open('x', encoding='utf-8') as stream:
        import json
        json.dump({'pid': os.getpid(), 'created': psutil.Process().create_time()}, stream)
    logger = RunLogger(group / 'logs')
    try:
        print('GROUP_ID=' + group_id, flush=True)
        print('SELECTED_FOLDS=' + ','.join(map(str, selected)), flush=True)
        cache = group / 'window_index' if source else cache_path(config)
        path = group / 'experiment_manifest.json'
        if resume:
            manifest = existing
        else:
            manifest = {'group_id': group_id, 'effective_config': config, 'manifest_checksum': digest(split),
                        'cache_path': str(cache), 'status': 'running',
                        'runner_version': 'tensorboard-manual-v1', 'requested_folds': selected}
            if source:
                notebook = 'notebooks/02_preprocessing_common_rate.ipynb'
                manifest.update(data_source=source, preprocessing_notebook={'path': notebook, 'sha256': file_hash(ROOT / notebook)})
            else:
                manifest['raw_root'] = str(raw_root())
            write_json(path, manifest)
            write_json(group / 'split_manifest.json', split)
            (group / 'split_manifest.sha256').write_text(digest(split) + '\n', encoding='ascii')
            (group / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
            system = system_info()
            write_json(group / 'system.json', system)
            (group / 'requirements-runtime.txt').write_text('\n'.join(f'{name}=={version}' for name, version in system['packages'].items() if version is not None) + '\n', encoding='utf-8')
            source_files = [p for directory in ['src', 'scripts', 'configs', 'experiments'] for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml')]
            for source in source_files:
                destination = group / 'source' / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            write_json(group / 'source_checksums.json', {p.relative_to(ROOT).as_posix(): file_hash(p) for p in source_files})
        logger.event('experiment_start', group_id=group_id, resume=resume)
        if source:
            audit, rows = build_index(source, config['data'], split['original_subject_order'])
            if resume and manifest.get('dataset_fingerprint', audit['dataset_fingerprint']) != audit['dataset_fingerprint']:
                raise ValueError('Common-rate notebook files changed since the original experiment.')
            prepare_index(cache, audit, rows)
            caches = audit['subjects']
            manifest.update(dataset_fingerprint=audit['dataset_fingerprint'], window_count=audit['windows'],
                            window_manifest='window_index/windows.csv', data_audit='window_index/common_rate_index.json')
            logger.event('window_index_ready', root=source, index=str(cache), windows=audit['windows'],
                         rejected_quality_windows=audit['rejected_quality_windows'])
        else:
            logger.event('preprocessing', root=str(raw_root()), cache=str(cache))
            caches = prepare_dataset(raw_root(), cache, config)
        cache_keys = {m['subject_id']: m['cache_key'] for m in caches}
        if resume and 'cache_keys' in manifest and manifest['cache_keys'] != cache_keys:
            raise ValueError('Raw dataset/cache differs from the original experiment.')
        manifest['cache_keys'] = cache_keys
        manifest.update(status='running', requested_folds=selected)
        manifest.pop('error', None)
        write_group_manifest(group, manifest)
        completed = []
        for seed in config['training']['seeds']:
            for fold in split['folds']:
                if fold['fold_id'] not in selected:
                    continue
                completed.append(train_fold(group / f'seed_{seed}' / f'fold_{fold["fold_id"]:02d}', cache, fold, seed, config, digest(split), resume=resume, group_id=group_id, total_folds=len(split['folds'])))
        # New reproduction groups get their own outputs. Historical README result
        # blocks are deliberately not rewritten by manual reproduction runs.
        result = aggregate_group(group, update_docs=False)
        manifest['status'] = result['status']
        write_group_manifest(group, manifest)
        generate_visualizations(group_id, root=ROOT)
        logger.event('experiment_end', status=manifest['status'], completed_folds=result['completed_folds'])
        ConsoleLogger(config.get('logging', {}).get('console')).cv_summary(completed)
        print('GROUP_ID=' + group_id, flush=True)
        return group
    except BaseException as exc:
        logger.error(exc)
        if (group / 'experiment_manifest.json').exists():
            manifest = read_json(group / 'experiment_manifest.json')
            manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc))
            write_group_manifest(group, manifest)
        if experiment_path is None:
            experiment_path = group / 'config.yaml'
        print(f'Resume: python -m scripts.run_grouped_cv --experiment {experiment_path} --resume --group-id {group_id}', flush=True)
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='experiments/e1_cnn/config.yaml')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--group-id')
    parser.add_argument('--folds', type=int, nargs='+', help='Official fold IDs to execute; default all five. Resume defaults to the saved selection.')
    parser.add_argument('--dry-run', action='store_true', help='Verify config, splits and input files without training or writing artifacts.')
    parser.add_argument('--tracking', choices=['local', 'tensorboard'])
    parser.add_argument('--non-interactive', action='store_true')
    args = parser.parse_args()
    config = load_config(args.experiment)
    if args.tracking:
        config['tracking']['backend'] = args.tracking
        config['tracking']['tensorboard']['enabled'] = args.tracking == 'tensorboard'
    if args.non_interactive:
        config['logging']['console']['progress'] = 'never'
    run(config, args.group_id, args.resume, folds=args.folds, dry_run=args.dry_run, experiment_path=args.experiment)


if __name__ == '__main__':
    main()
