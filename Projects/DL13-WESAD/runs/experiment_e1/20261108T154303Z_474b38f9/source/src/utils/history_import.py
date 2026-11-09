"""Archived W&B replay tool, never called by active TensorBoard training."""
import os
from pathlib import Path
import psutil
from src.evaluation.history import read_group
from src.utils.config import ROOT, merge
from src.utils.tracking import digest, read_json, write_json
from src.utils.wandb_tracker import WandbTracker, redact


def import_history(group_id, mode='offline', dry_run=False, root=ROOT, overrides=None, sdk=None):
    if mode not in ('offline', 'online'):
        raise ValueError('Import mode must be explicitly offline or online.')
    history = read_group(group_id, root)
    # Archived, explicit-only importer; independent of the active TensorBoard schema.
    tracking = {'backend': 'wandb', 'strict': True, 'wandb': {'enabled': True, 'mode': mode,
        'project': 'DL13-WESAD', 'entity': None, 'batch_log_frequency': 10,
        'log_epoch_metrics': True, 'log_batch_metrics': True, 'log_system_metrics': False,
        'log_figures': True, 'log_confusion_matrix': True, 'upload_checkpoints': False,
        'upload_raw_data': False}}
    config = merge(history['manifest']['effective_config'], {'tracking': tracking})
    config = merge(config, overrides or {})
    # Historical import is metrics/derived figures only, even if future training enables artifacts.
    config['tracking']['wandb'].update(enabled=True, mode=mode, upload_checkpoints=False, upload_raw_data=False, log_system_metrics=False)
    frequency = config['tracking']['wandb']['batch_log_frequency']
    plan = {'group_id': group_id, 'mode': mode, 'historical_import': True, 'dry_run': dry_run, 'folds': [], 'imported': 0, 'skipped': 0}
    for fold in history['folds']:
        plan['folds'].append({'fold_id': fold.metadata['fold_id'], 'local_run_id': fold.metadata['run_id'],
            'training_seed': fold.metadata['training_seed'], 'epochs': len(fold.epochs), 'committed_batches': len(fold.iterations),
            'selected_wandb_batches': sum(int(r['global_step']) % frequency == 0 for r in fold.iterations),
            'missing_metrics': [key for key in ('train_accuracy', 'val_accuracy') if not any(r.get(key) is not None for r in fold.epochs)]})
    if dry_run:
        return plan
    output = history['result'] / 'wandb_imports'
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.import.lock'
    if lock.exists():
        previous = read_json(lock)
        try:
            alive = abs(psutil.Process(previous['pid']).create_time() - previous['created']) < .01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError('A historical import is already active.')
        lock.unlink()
    with lock.open('x', encoding='utf-8') as stream:
        import json
        json.dump({'pid': os.getpid(), 'created': psutil.Process().create_time()}, stream)
    registry_path = output / 'import_registry.json'
    registry = read_json(registry_path) if registry_path.exists() else {'version': 1, 'imports': {}}
    try:
        for fold, item in zip(history['folds'], plan['folds']):
            wb = config['tracking']['wandb']
            key = digest({'local_run_id': fold.metadata['run_id'], 'group_id': group_id, 'project': wb['project'], 'entity': wb['entity'], 'mode': mode})
            source_id = digest(history['source_hashes'])
            receipt = output / 'receipts' / (key + '.json')
            existing = registry['imports'].get(key)
            if existing:
                if existing['source_digest'] != source_id:
                    raise ValueError('Source history changed after an earlier import; refusing duplicate replay.')
                if receipt.exists() and read_json(receipt)['source_digest'] == source_id:
                    existing.update(read_json(receipt))
                    registry['imports'][key] = existing
                    write_json(registry_path, registry)
                if existing['status'] == 'completed':
                    item.update(status='skipped', wandb_run_id=existing['wandb_run_id'], directory=existing.get('directory'))
                    plan['skipped'] += 1
                    continue
                raise RuntimeError('A prior import is incomplete. Inspect the registry/offline directory before retrying; automatic replay could duplicate history.')
            entry = {'local_run_id': fold.metadata['run_id'], 'wandb_run_id': key[:24], 'source_digest': source_id, 'mode': mode, 'status': 'in_progress'}
            registry['imports'][key] = entry
            write_json(registry_path, registry)
            tracker = WandbTracker(config, output / mode, sdk=sdk)
            try:
                metadata = {**fold.metadata, 'effective_config': config}
                identity = tracker.start_run(metadata, group_id, run_id=entry['wandb_run_id'], historical=True)
                if identity is None:
                    raise RuntimeError('W&B did not initialize; no import was completed.')
                entry.update(identity)
                write_json(registry_path, registry)
                validations = {r['epoch']: r for r in fold.validation}
                for row in fold.epochs:
                    for batch in fold.iterations:
                        if batch['epoch'] == row['epoch']:
                            tracker.log_batch(batch)
                    tracker.log_epoch(row)
                    tracker.log_validation(validations[row['epoch']])
                tracker.log_checkpoint(fold.folder / 'checkpoints/best.pt', fold.metadata['best_epoch'], fold.metadata['best_val_macro_f1'])
                tracker.log_test(fold.test, sorted((fold.folder / 'plots').glob('*.png')))
                tracker.finish_run('completed')
                entry['status'] = 'completed'
                write_json(receipt, entry)
                write_json(registry_path, registry)
                item.update(status='imported', **identity)
                plan['imported'] += 1
            except BaseException as exc:
                entry.update(status='failed', error=redact(f'{type(exc).__name__}: {exc}'))
                write_json(registry_path, registry)
                raise
            finally:
                tracker.finish_run(entry['status'])
        write_json(output / 'last_import.json', plan)
        return plan
    finally:
        lock.unlink(missing_ok=True)
