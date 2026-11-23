"""Archived W&B adapter. Not imported by the active TensorBoard training pipeline."""
from contextlib import contextmanager
import importlib
import netrc
import os
from pathlib import Path
import re
from urllib.parse import urlparse
import warnings
from src.utils.tracking import digest

SECRET = re.compile(r'api.?key|token|password|secret|credential|authorization', re.I)


def redact(value):
    if isinstance(value, dict):
        return {str(k): '[REDACTED]' if SECRET.search(str(k)) else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        for name, secret in os.environ.items():
            if SECRET.search(name) and len(secret) >= 6:
                value = value.replace(secret, '[REDACTED]')
    return value


def tracking_options(config):
    options = config.get('tracking', {})
    wb = options.get('wandb', {})
    if any(SECRET.search(key) for key in wb):
        raise ValueError('Credentials must be supplied through the environment or user credential store, never YAML.')
    if options.get('backend', 'local') not in ('local', 'wandb'):
        raise ValueError('tracking.backend must be local or wandb.')
    if wb.get('mode', 'offline') not in ('offline', 'online'):
        raise ValueError('W&B mode must be offline or explicitly online.')
    if wb.get('upload_raw_data', False):
        raise ValueError('Raw WESAD upload is not supported by the tracker.')
    if wb.get('batch_log_frequency', 10) < 1:
        raise ValueError('batch_log_frequency must be positive.')
    return options, wb


def has_online_auth():
    if os.environ.get('WANDB_API_KEY'):
        return True
    host = urlparse(os.environ.get('WANDB_BASE_URL', 'https://api.wandb.ai')).hostname
    for name in ('.netrc', '_netrc'):
        try:
            if netrc.netrc(str(Path.home() / name)).authenticators(host):
                return True
        except (OSError, netrc.NetrcParseError):
            pass
    return False


@contextmanager
def preserve_rng():
    # The existing training RNG implementation remains untouched.
    from src.utils.reproducibility import rng_state, restore_rng
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class WandbTracker:
    def __init__(self, config, directory, warning=None, sdk=None):
        options, self.options = tracking_options(config)
        self.enabled = options.get('backend', 'local') == 'wandb' and self.options.get('enabled', False)
        self.strict = options.get('strict', False)
        self.directory = Path(directory).resolve()
        self.warning = warning or (lambda message: warnings.warn(message, RuntimeWarning))
        self.sdk, self.run = sdk, None
        self.failed, self.finished = False, False

    @contextmanager
    def environment(self):
        """Keep SDK caches/staging in the run area; restore the caller's environment."""
        settings = {'WANDB_DATA_DIR': self.directory / 'sdk_data', 'WANDB_CACHE_DIR': self.directory / 'sdk_cache',
                    'WANDB_CONFIG_DIR': self.directory / 'sdk_config'}
        for path in settings.values():
            path.mkdir(parents=True, exist_ok=True)
        updates = {key: str(value) for key, value in settings.items()}
        updates['WANDB_MODE'] = self.options.get('mode', 'offline')
        previous = {key: os.environ.get(key) for key in updates}
        os.environ.update(updates)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def _call(self, action):
        if not self.enabled or self.failed:
            return None
        try:
            with preserve_rng(), self.environment():
                return action()
        except Exception as exc:
            self.failed = True
            message = redact(f'W&B tracking failed ({type(exc).__name__}): {exc}; local logging remains available.')
            self.warning(message)
            if self.strict:
                raise RuntimeError(message) from None
            return None

    def start_run(self, metadata, group_id, run_id=None, historical=False):
        def start():
            self.sdk = self.sdk or importlib.import_module('wandb')
            mode = self.options.get('mode', 'offline')
            if mode == 'online' and not has_online_auth():
                raise RuntimeError('Online mode requires existing credentials; authenticate yourself with wandb login first.')
            self.directory.mkdir(parents=True, exist_ok=True)
            local_id = metadata['run_id']
            attempt = metadata.get('attempt_id', metadata.get('attempts', [{}])[-1].get('attempt_id', 'initial'))
            # Offline SDK resume is unsupported: retain distinct linked attempt segments.
            identity = local_id if mode == 'online' else f'{local_id}/{attempt}'
            wb_id = run_id or digest(identity)[:24]
            settings = self.sdk.Settings(console='off', quiet=True, silent=True, disable_git=True,
                disable_code=True, save_code=False, disable_job_creation=True,
                x_disable_meta=True, x_disable_stats=not self.options.get('log_system_metrics', True))
            config = metadata['effective_config']
            exported = {k: metadata[k] for k in ('fold_id', 'training_seed', 'manifest_checksum', 'subjects', 'window_counts', 'trainable_parameters', 'device') if k in metadata}
            self.run = self.sdk.init(project=self.options.get('project', 'DL13-WESAD'), entity=self.options.get('entity'),
                name=f"{config['experiment']['id']}_{config['experiment']['model']}_fold{metadata['fold_id']:02d}_seed{metadata['training_seed']}",
                group=group_id, id=wb_id, job_type='historical_import' if historical else 'train', mode=mode,
                dir=str(self.directory), resume='allow' if mode == 'online' and not historical else None,
                reinit='create_new', save_code=False, settings=settings,
                tags=['historical_import'] if historical else ['training'],
                config=redact({'effective_config': config, 'local_run_id': local_id, 'attempt_id': attempt, 'group_id': group_id,
                               'historical_import': historical, **exported}))
            for axis in ('epoch', 'global_step'):
                self.run.define_metric(axis)
            for namespace in ('train/*', 'val/*', 'optimization/*', 'selection/*'):
                self.run.define_metric(namespace, step_metric='epoch')
            self.run.define_metric('batch/*', step_metric='global_step')
            self.run.summary.update({'local_run_id': local_id, 'historical_import': historical, 'mode': mode})
            return {'wandb_run_id': self.run.id, 'directory': self.run.dir, 'mode': mode}
        return self._call(start)

    def log_batch(self, row):
        if self.options.get('log_batch_metrics', True) and int(row['global_step']) % self.options.get('batch_log_frequency', 10) == 0:
            data = {'global_step': int(row['global_step']), 'batch/epoch': int(row['epoch']), 'batch/attempt_id': row.get('attempt_id', '')}
            for source, key in [('train_loss', 'loss'), ('learning_rate', 'learning_rate'), ('gradient_norm', 'gradient_norm'), ('batch_duration_ms', 'duration_ms')]:
                if row.get(source) is not None:
                    data['batch/' + key] = row[source]
            self._call(lambda: self.run.log(data))

    def log_epoch(self, row):
        if not self.options.get('log_epoch_metrics', True):
            return
        data = {'epoch': int(row['epoch'])}
        keys = {'train_online_loss': 'train/online_loss', 'train_eval_loss': 'train/loss',
                'train_accuracy': 'train/accuracy', 'train_macro_f1': 'train/macro_f1', 'train_balanced_accuracy': 'train/balanced_accuracy',
                'val_loss': 'val/loss', 'val_accuracy': 'val/accuracy', 'val_macro_f1': 'val/macro_f1', 'val_balanced_accuracy': 'val/balanced_accuracy',
                'learning_rate': 'optimization/learning_rate', 'epoch_duration_seconds': 'optimization/epoch_duration_seconds',
                'best_epoch': 'selection/best_epoch', 'best_val_macro_f1': 'selection/best_val_macro_f1'}
        data.update({key: row[source] for source, key in keys.items() if row.get(source) is not None})
        self._call(lambda: self.run.log(data))

    def log_validation(self, row):
        data = {'epoch': int(row['epoch'])}
        for name in ('precision', 'recall', 'f1'):
            for index, value in enumerate(row.get(f'val_{name}_per_class') or []):
                data[f'val/class_{index}/{name}'] = value
        if len(data) > 1:
            self._call(lambda: self.run.log(data))

    def log_checkpoint(self, path, epoch, score):
        def log():
            self.run.summary.update({'selection/best_epoch': epoch, 'selection/best_val_macro_f1': score})
            if self.options.get('upload_checkpoints', False):
                artifact = self.sdk.Artifact(f'{self.run.id}-checkpoint', type='model')
                artifact.add_file(str(path))
                self.run.log_artifact(artifact)
        self._call(log)

    def log_test(self, metrics, figures=None):
        def log():
            scores = {'test/' + key: metrics[key] for key in ('loss', 'accuracy', 'macro_f1', 'balanced_accuracy') if key in metrics}
            self.run.log(scores)
            self.run.summary.update(scores)
            if self.options.get('log_confusion_matrix', True) and 'confusion_matrix' in metrics:
                names = metrics.get('label_order', [str(i) for i in range(len(metrics['confusion_matrix']))])
                rows = [[names[i], names[j], count] for i, row in enumerate(metrics['confusion_matrix']) for j, count in enumerate(row)]
                table = self.sdk.Table(columns=['Actual', 'Predicted', 'Count'], data=rows)
                self.run.log({'test/confusion_matrix': self.sdk.plot_table('wandb/confusion_matrix/v1', table,
                               {'Actual': 'Actual', 'Predicted': 'Predicted', 'nPredictions': 'Count'})})
            if self.options.get('log_figures', True):
                for path in figures or []:
                    self.run.log({'figures/' + Path(path).stem: self.sdk.Image(str(path))})
        self._call(log)

    def finish_run(self, status='completed'):
        if self.finished:
            return
        self.finished = True
        if self.run is not None:
            try:
                with preserve_rng(), self.environment():
                    self.run.summary['status'] = status
                    self.run.finish(exit_code=0 if status == 'completed' and not self.failed else 1)
            except Exception as exc:
                message = redact(f'W&B finish failed ({type(exc).__name__}): {exc}')
                self.warning(message)
                if self.strict:
                    raise RuntimeError(message) from None
