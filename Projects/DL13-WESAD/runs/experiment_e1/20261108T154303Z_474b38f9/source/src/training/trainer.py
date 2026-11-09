"""Train a fold, recover at epoch boundaries, select exclusively on validation."""
import copy
import time
import warnings
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from src.data.dataset import WindowDataset
from src.data.normalization import ChannelScaler
from src.models.cnn import CNN1D
from src.training.optimizer import make_optimizer, class_weights
from src.training.callbacks import SelectionState
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.evaluation.evaluator import evaluate, export_test
from src.evaluation.visualization import training_plots, confusion_plot
from src.utils.experiment_tracking import TrainingTracker
from src.utils.config import same_training_config
from src.utils.reproducibility import seed_everything, rng_state, restore_rng
from src.utils.tracking import now, identifier, write_json, read_json, file_hash


def completed_fold(folder):
    marker = folder / 'completion.json'
    if not marker.exists():
        return False
    recorded = read_json(marker)
    if not all((folder / f).is_file() and file_hash(folder / f) == h for f, h in recorded['files'].items()):
        raise ValueError(f'Completed fold artifacts were modified or are missing: {folder}')
    return read_json(folder / 'metadata.json')['status'] == 'completed'


def recover_epoch_logs(logger, history):
    """Repair the small crash interval after checkpoint commit but before log append."""
    for field, method in [('epoch_log', logger.epoch), ('validation_log', logger.validation)]:
        for record in history:
            row = record[field]
            # Idempotent writes also reconcile a missing v2 sidecar after a crash.
            method(row)


def train_fold(folder, cache, fold, seed, config, manifest_checksum, resume=False, interrupt=None, group_id=None, total_folds=None):
    folder.mkdir(parents=True, exist_ok=True)
    if completed_fold(folder):
        if not resume:
            raise FileExistsError('Completed fold exists; use resume to skip it.')
        print(f'[SKIP] seed={seed} fold={fold["fold_id"]}', flush=True)
        return read_json(folder / 'metadata.json')
    logger = TrainingTracker(folder / 'logs', config)
    attempt = identifier()
    metadata_path = folder / 'metadata.json'
    metadata = read_json(metadata_path) if metadata_path.exists() else {
        'run_id': identifier(), 'fold_id': fold['fold_id'], 'training_seed': seed,
        'subjects': fold, 'started_at': now(), 'manifest_checksum': manifest_checksum,
        'effective_config': config, 'attempts': []}
    if not same_training_config(metadata['effective_config'], config) or metadata['manifest_checksum'] != manifest_checksum or metadata['subjects'] != fold or metadata['training_seed'] != seed:
        raise ValueError('Resume metadata/config/split/seed mismatch.')
    metadata['attempts'].append({'attempt_id': attempt, 'started_at': now()})
    metadata['status'] = 'running'
    metadata['effective_config'] = config
    write_json(metadata_path, metadata)
    started = time.perf_counter()
    history, elapsed_before = [], 0.0
    try:
        seed_everything(seed, config['runtime']['deterministic'], config['runtime']['torch_threads'])
        requested = config['runtime']['device']
        device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)
        logger.event('fold_start', run_id=metadata['run_id'], attempt_id=attempt, fold_id=fold['fold_id'], seed=seed, device=str(device))
        datasets = {role: WindowDataset(cache, fold[role]) for role in ('train', 'validation', 'test')}
        scaler = ChannelScaler().fit(datasets['train'])
        scaler.save(folder / 'normalization.json')
        for dataset in datasets.values():
            dataset.scaler = scaler
        logger.event('preprocessing', train_subjects=fold['train'], validation_subjects=fold['validation'], test_subjects=fold['test'], window_counts={k: len(v) for k, v in datasets.items()})
        tc = config['training']
        loaders = {role: DataLoader(dataset, batch_size=tc['batch_size'], shuffle=False, num_workers=config['runtime']['num_workers']) for role, dataset in datasets.items()}
        train_loader = DataLoader(datasets['train'], batch_size=tc['batch_size'], shuffle=True, num_workers=config['runtime']['num_workers'])
        model = CNN1D(datasets['train'].arrays[fold['train'][0]].shape[1]).to(device)
        optimizer = make_optimizer(model, tc)
        weights = class_weights(datasets['train'].labels).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        selection = SelectionState(tc['early_stopping_patience'], tc['early_stopping_min_delta'])
        last_path, best_path = folder / 'checkpoints/last.pt', folder / 'checkpoints/best.pt'
        epoch_start, step, best_state = 1, 0, None
        cache_keys = {s: key for dataset in datasets.values() for s, key in dataset.cache_keys.items()}
        if last_path.exists():
            if not resume:
                raise FileExistsError('Checkpoint exists; explicit resume required.')
            state = load_checkpoint(last_path)
            if not same_training_config(state['effective_config'], config) or state['manifest_checksum'] != manifest_checksum or state['cache_keys'] != cache_keys or state['run_id'] != metadata['run_id']:
                raise ValueError('Checkpoint identity/config/cache checksum mismatch.')
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            selection.__dict__.update(state['early_stopping'])
            epoch_start, step = state['epoch'] + 1, state['global_step']
            history, best_state, elapsed_before = state['history'], state['best_model'], state['training_duration_seconds']
            # last.pt is authoritative even if an interrupted future epoch wrote best.pt.
            save_checkpoint(best_path, {**state, 'model': best_state, 'epoch': selection.best_epoch, 'checkpoint_kind': 'best'})
            recover_epoch_logs(logger, history)
            restore_rng(state['rng'])
            logger.event('resume', epoch_start=epoch_start, committed_global_step=step, attempt_id=attempt)
        elif resume:
            logger.event('resume', epoch_start=1, reason='No committed epoch; replay with a new attempt ID', attempt_id=attempt)
        metadata.update(logging_schema_version=2, device=str(device), architecture=str(model), trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        window_counts={k: len(v) for k, v in datasets.items()},
                        label_distribution={k: np.bincount(v.labels, minlength=3).tolist() for k, v in datasets.items()},
                        class_weights=weights.cpu().tolist(), cache_keys=cache_keys)
        (folder / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        write_json(metadata_path, metadata)
        logger.event('model', parameters=metadata['trainable_parameters'])
        logger.start_run(metadata, group_id or folder.parent.parent.name, total_folds or fold['fold_id'],
                         {'history': history, 'epoch': epoch_start - 1, 'global_step': step} if resume else None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', RuntimeWarning)
            for epoch in range(epoch_start, tc['max_epochs'] + 1):
                if selection.bad_epochs >= selection.patience:
                    break
                epoch_time = time.perf_counter()
                logger.event('epoch_start', epoch=epoch, attempt_id=attempt)
                logger.start_epoch(epoch, tc['max_epochs'], len(train_loader))
                model.train()
                numerator, denominator = 0.0, 0.0
                for batch_index, (x, y, _) in enumerate(train_loader):
                    batch_time = time.perf_counter()
                    x, y = x.to(device), y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(x)
                    loss = criterion(logits, y)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('NaN/Inf training loss')
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['gradient_clip_norm'], error_if_nonfinite=True)
                    optimizer.step()
                    step += 1
                    weight_sum = weights[y].sum().item()
                    numerator += loss.item() * weight_sum
                    denominator += weight_sum
                    logger.log_batch({'run_id': metadata['run_id'], 'attempt_id': attempt, 'fold_id': fold['fold_id'], 'training_seed': seed,
                                      'epoch': epoch, 'batch_index': batch_index, 'global_step': step, 'batch_size': len(y),
                                      'train_loss': loss.item(), 'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm': norm.item(),
                                      'batch_duration_ms': (time.perf_counter() - batch_time) * 1000, 'timestamp': now()})
                    if interrupt:
                        interrupt(epoch, batch_index)
                train_metrics, _, _ = evaluate(model, loaders['train'], device, weights)
                val, _, _ = evaluate(model, loaders['validation'], device, weights)
                better, stop = selection.update(val['macro_f1'], val['loss'], epoch)
                if better:
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epoch_row = {'attempt_id': attempt, 'epoch': epoch, 'train_online_loss': numerator / denominator, 'train_eval_loss': train_metrics['loss'],
                             'early_stopping_counter': selection.bad_epochs,
                             'train_accuracy': train_metrics['accuracy'], 'val_accuracy': val['accuracy'],
                             'train_macro_f1': train_metrics['macro_f1'], 'train_balanced_accuracy': train_metrics['balanced_accuracy'],
                             'val_loss': val['loss'], 'val_macro_f1': val['macro_f1'], 'val_balanced_accuracy': val['balanced_accuracy'],
                             'learning_rate': optimizer.param_groups[0]['lr'], 'epoch_duration_seconds': time.perf_counter() - epoch_time,
                             'best_epoch': selection.best_epoch, 'best_val_macro_f1': selection.best_score}
                val_row = {'attempt_id': attempt, 'epoch': epoch, 'val_loss': val['loss'], 'val_macro_f1': val['macro_f1'], 'val_balanced_accuracy': val['balanced_accuracy'],
                           'val_accuracy': val['accuracy'],
                           'val_precision_per_class': val['precision_per_class'], 'val_recall_per_class': val['recall_per_class'], 'val_f1_per_class': val['f1_per_class'], 'is_best_checkpoint': better}
                history.append({**epoch_row, 'epoch_log': epoch_row, 'validation_log': val_row})
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': None, 'epoch': epoch,
                         'best_epoch': selection.best_epoch, 'best_validation_score': selection.best_score, 'early_stopping': copy.deepcopy(selection.__dict__),
                         'rng': rng_state(), 'effective_config': config, 'manifest_checksum': manifest_checksum, 'cache_keys': cache_keys,
                         'run_id': metadata['run_id'], 'attempt_id': attempt, 'global_step': step, 'history': history, 'best_model': best_state,
                         'training_duration_seconds': elapsed_before + time.perf_counter() - started, 'checkpoint_kind': 'last'}
                if better:
                    save_checkpoint(best_path, {**state, 'model': best_state, 'checkpoint_kind': 'best'})
                save_checkpoint(last_path, state)
                logger.log_epoch(epoch_row, val_row, {'is_best': better, 'bad_epochs': selection.bad_epochs,
                    'elapsed': elapsed_before + time.perf_counter() - started})
                if better:
                    logger.log_checkpoint(best_path, selection.best_epoch, selection.best_score)
                logger.event('validation_completed', epoch=epoch, val_macro_f1=val['macro_f1'], val_loss=val['loss'])
                logger.event('checkpoint_saved', epoch=epoch, best=better)
                logger.event('epoch_end', epoch=epoch, **{k: epoch_row[k] for k in ('train_online_loss', 'val_macro_f1', 'best_epoch')})
                if stop:
                    logger.event('early_stopping', epoch=epoch, patience=selection.patience)
                    break
            if not history:
                raise ValueError('No completed training epochs.')
            # Test loader is first consumed only after model selection has finished.
            model.load_state_dict(load_checkpoint(best_path)['model'])
            test, probabilities, indices = evaluate(model, loaders['test'], device, weights)
            export_test(folder / 'metrics', datasets['test'], test, probabilities, indices, fold['fold_id'])
            write_json(folder / 'metrics/best_validation.json', history[selection.best_epoch - 1]['validation_log'])
            training_plots(history, folder / 'plots')
            confusion_plot(test['confusion_matrix'], folder / 'plots/confusion_matrix.png')
            for warning in caught:
                with (logger.folder / 'errors.log').open('a', encoding='utf-8') as stream:
                    stream.write(f'{now()} WARNING {warning.message}\n')
        metadata.update(status='completed', ended_at=now(), completed_epochs=len(history), best_epoch=selection.best_epoch,
                        best_val_macro_f1=selection.best_score, test_metrics=test,
                        training_duration_seconds=elapsed_before + time.perf_counter() - started,
                        checkpoints={'best': str(best_path), 'last': str(last_path)})
        write_json(metadata_path, metadata)
        logger.event('test_completed', macro_f1=test['macro_f1'], balanced_accuracy=test['balanced_accuracy'])
        logger.event('fold_end', status='completed')
        logger.log_test(test, metadata, sorted((folder / 'plots').glob('*.png')))
        logger.finish_run('completed')
        files = [p for p in folder.rglob('*') if p.is_file() and p.name != 'completion.json' and '.tmp' not in p.name]
        write_json(folder / 'completion.json', {'files': {p.relative_to(folder).as_posix(): file_hash(p) for p in files}})
        return metadata
    except BaseException as exc:
        metadata.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=repr(exc), ended_at=now())
        write_json(metadata_path, metadata)
        logger.error(exc)
        raise
    finally:
        logger.finish_run(metadata.get('status', 'failed'))
