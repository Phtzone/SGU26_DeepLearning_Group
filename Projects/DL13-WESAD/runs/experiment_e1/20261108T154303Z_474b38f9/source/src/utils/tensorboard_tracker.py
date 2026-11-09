"""Optional, local-only SummaryWriter backend with committed-history recovery."""
from contextlib import contextmanager
import csv
import json
import math
from pathlib import Path
import uuid
import warnings


def tracking_options(config):
    options = config.get('tracking', {})
    tb = options.get('tensorboard', {})
    if options.get('backend', 'local') not in ('local', 'tensorboard'):
        raise ValueError('Active tracking backends are local and tensorboard; W&B is retired.')
    if not options.get('local', {}).get('enabled', True):
        raise ValueError('Local CSV/JSON logging must remain enabled.')
    for key, default in [('flush_secs', 10), ('max_queue', 10), ('batch_log_frequency', 1)]:
        if not isinstance(tb.get(key, default), (int, float)) or tb.get(key, default) <= 0:
            raise ValueError(f'tracking.tensorboard.{key} must be positive.')
    return options, tb


@contextmanager
def preserve_rng():
    from src.utils.reproducibility import rng_state, restore_rng
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class TensorBoardTracker:
    def __init__(self, config, directory, warning=None, writer_factory=None):
        options, self.options = tracking_options(config)
        self.enabled = options.get('backend', 'local') == 'tensorboard' and self.options.get('enabled', False)
        self.strict = options.get('strict', False)
        self.directory = Path(directory)
        self.warning = warning or (lambda text: warnings.warn(text, RuntimeWarning))
        self.writer_factory = writer_factory
        self.writers = {}
        self.failed = self.closed = False
        self.last_batch_step = self.last_epoch = self.last_validation_epoch = 0
        self.metadata = {}

    def _call(self, action):
        if not self.enabled or self.failed or self.closed:
            return None
        try:
            with preserve_rng():
                return action()
        except Exception as exc:
            self.failed = True
            message = f'TensorBoard tracking failed ({type(exc).__name__}): {exc}; local logs are preserved.'
            self.warning(message)
            if self.strict:
                raise RuntimeError(message) from None
            return None

    def start_run(self, metadata, group_id, resume_state=None):
        def start():
            if self.writer_factory is None:
                from torch.utils.tensorboard import SummaryWriter
                self.writer_factory = SummaryWriter
            self.metadata = metadata
            self.directory.mkdir(parents=True, exist_ok=True)
            resuming = resume_state is not None
            if not resuming and list(self.directory.rglob('events.out.tfevents.*')):
                raise FileExistsError('Existing TensorBoard events require explicit resume.')
            for stream in ('epochs', 'batches', 'evaluation'):
                self.writers[stream] = self.writer_factory(log_dir=str(self.directory / stream),
                    purge_step=0 if resuming else None, flush_secs=self.options.get('flush_secs', 10),
                    max_queue=self.options.get('max_queue', 10), filename_suffix='.' + uuid.uuid4().hex[:12])
            self.writers['epochs'].add_text('run/identity', json.dumps({'group_id': group_id,
                'local_run_id': metadata['run_id'], 'fold_id': metadata['fold_id'],
                'training_seed': metadata['training_seed'], 'manifest_checksum': metadata['manifest_checksum']}, sort_keys=True), 0)
            self.writers['epochs'].add_text('run/effective_config', json.dumps(metadata['effective_config'], indent=2), 0)
            self.writers['epochs'].add_custom_scalars({'Learning curves': {
                'Loss': ['Multiline', ['train/eval_loss', 'val/loss', 'train/online_loss']],
                'Accuracy': ['Multiline', ['train/accuracy', 'val/accuracy']],
                'Macro-F1': ['Multiline', ['train/macro_f1', 'val/macro_f1']],
                'Balanced accuracy': ['Multiline', ['train/balanced_accuracy', 'val/balanced_accuracy']]}})
            if resuming:
                # Purge only this fold's event streams, then replay authoritative committed
                # records. Epoch and batch axes stay separate; partial-epoch points vanish.
                history = resume_state.get('history', [])
                committed = {(str(row['attempt_id']), int(row['epoch'])) for row in history}
                csv_path = self.directory.parent / 'logs/iterations.csv'
                if csv_path.exists():
                    with csv_path.open(encoding='utf-8', newline='') as stream:
                        for row in csv.DictReader(stream):
                            if (row['attempt_id'], int(row['epoch'])) in committed:
                                parsed = {key: float(value) if key in ('train_loss', 'learning_rate', 'gradient_norm', 'batch_duration_ms') else value for key, value in row.items()}
                                self._batch(parsed)
                for row in history:
                    self._epoch(row)
                    self._validation(row['validation_log'])
            for writer in self.writers.values():
                writer.flush()
            return {'backend': 'tensorboard', 'directory': str(self.directory),
                    'streams': {name: str(self.directory / name) for name in self.writers},
                    'run_id': metadata['run_id'], 'group_id': group_id}
        return self._call(start)

    @staticmethod
    def _scalar(writer, tag, value, step):
        if value is not None and math.isfinite(float(value)):
            writer.add_scalar(tag, float(value), int(step))

    def _batch(self, row):
        step = int(row['global_step'])
        if step <= self.last_batch_step:
            raise ValueError('Batch global_step must increase strictly within a committed session.')
        self.last_batch_step = step
        if not self.options.get('log_every_batch', True) or step % self.options.get('batch_log_frequency', 1):
            return
        for source, name in [('train_loss', 'loss'), ('learning_rate', 'learning_rate'),
                             ('gradient_norm', 'gradient_norm'), ('batch_duration_ms', 'duration_ms')]:
            self._scalar(self.writers['batches'], 'batch/' + name, row.get(source), step)

    def log_batch(self, row):
        return self._call(lambda: self._batch(row))

    def _epoch(self, row):
        epoch = int(row['epoch'])
        if epoch <= self.last_epoch:
            raise ValueError('Epoch step must increase strictly.')
        self.last_epoch = epoch
        if not self.options.get('log_every_epoch', True):
            return
        mapping = {'train_online_loss': 'train/online_loss', 'train_eval_loss': 'train/eval_loss',
            'train_accuracy': 'train/accuracy', 'train_macro_f1': 'train/macro_f1', 'train_balanced_accuracy': 'train/balanced_accuracy',
            'val_loss': 'val/loss', 'val_accuracy': 'val/accuracy', 'val_macro_f1': 'val/macro_f1', 'val_balanced_accuracy': 'val/balanced_accuracy',
            'learning_rate': 'optimization/learning_rate', 'epoch_duration_seconds': 'optimization/epoch_duration_seconds',
            'best_epoch': 'selection/best_epoch', 'best_val_macro_f1': 'selection/best_val_macro_f1',
            'early_stopping_counter': 'selection/early_stopping_counter'}
        for source, tag in mapping.items():
            self._scalar(self.writers['epochs'], tag, row.get(source), epoch)

    def log_epoch(self, row):
        return self._call(lambda: self._epoch(row))

    def _validation(self, row):
        epoch = int(row['epoch'])
        if epoch <= self.last_validation_epoch:
            raise ValueError('Validation epoch step must increase strictly.')
        self.last_validation_epoch = epoch
        if self.options.get('log_every_epoch', True):
            for metric in ('precision', 'recall', 'f1'):
                for index, value in enumerate(row.get(f'val_{metric}_per_class') or []):
                    self._scalar(self.writers['epochs'], f'val/class_{index}/{metric}', value, epoch)

    def log_validation(self, row):
        return self._call(lambda: self._validation(row))

    def log_checkpoint(self, path, epoch, score):
        return self._call(lambda: self.writers['epochs'].add_text('checkpoint/best', str(path), int(epoch)))

    def log_test(self, metrics, figures=None, metadata=None):
        def log():
            from src.evaluation.visualization import plt
            import numpy as np
            state = metadata or self.metadata
            step = int(state.get('completed_epochs', self.last_epoch))
            writer = self.writers['evaluation']
            for key in ('loss', 'accuracy', 'macro_f1', 'balanced_accuracy'):
                self._scalar(writer, 'test/' + key, metrics.get(key), step)
            if not self.options.get('log_figures', True):
                return
            labels = [label.title() for label in metrics.get('label_order', [])]
            if 'confusion_matrix' in metrics:
                matrix = np.asarray(metrics['confusion_matrix'])
                labels = labels or [str(i) for i in range(len(matrix))]
                for normalized in (False, True):
                    totals = matrix.sum(axis=1, keepdims=True)
                    values = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals != 0) if normalized else matrix
                    fig, ax = plt.subplots(figsize=(6, 5))
                    try:
                        ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
                        ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                               xlabel='Predicted class', ylabel='True class', title='Row-normalized confusion matrix' if normalized else 'Confusion matrix')
                        for i, row in enumerate(values):
                            for j, value in enumerate(row):
                                ax.text(j, i, f'{value:.1%}' if normalized else str(int(value)), ha='center', va='center')
                        fig.tight_layout()
                        writer.add_figure('test/confusion_normalized' if normalized else 'test/confusion_raw', fig, step, close=True)
                    finally:
                        plt.close(fig)
            if 'f1_per_class' in metrics:
                fig, ax = plt.subplots(figsize=(6, 4))
                try:
                    ax.bar(labels, metrics['f1_per_class'], color='#0072B2')
                    ax.set(ylabel='F1 score', ylim=(0, 1), title='Test F1 by class')
                    fig.tight_layout()
                    writer.add_figure('test/per_class_f1', fig, step, close=True)
                finally:
                    plt.close(fig)
        return self._call(log)

    def flush(self):
        return self._call(lambda: [writer.flush() for writer in self.writers.values()])

    def finish_run(self, status='completed'):
        if self.closed:
            return
        self.closed = True
        errors = []
        with preserve_rng():
            for writer in self.writers.values():
                try:
                    writer.flush()
                except Exception as exc:
                    errors.append(exc)
                finally:
                    try:
                        writer.close()
                    except Exception as exc:
                        errors.append(exc)
        if errors:
            message = f'TensorBoard cleanup failed: {type(errors[0]).__name__}: {errors[0]}'
            self.warning(message)
            if self.strict:
                raise RuntimeError(message) from None
