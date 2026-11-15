"""Append-only CSV and event logs, flushed on every record."""
import csv
import json
import traceback
from pathlib import Path
from src.utils.tracking import now

ITERATION_FIELDS = ['run_id', 'attempt_id', 'fold_id', 'training_seed', 'epoch', 'batch_index', 'global_step', 'batch_size', 'train_loss', 'learning_rate', 'gradient_norm', 'batch_duration_ms', 'timestamp']
LEGACY_EPOCH_FIELDS = ['attempt_id', 'epoch', 'train_online_loss', 'train_eval_loss', 'train_macro_f1', 'train_balanced_accuracy', 'val_loss', 'val_macro_f1', 'val_balanced_accuracy', 'learning_rate', 'epoch_duration_seconds', 'best_epoch', 'best_val_macro_f1']
LEGACY_VALIDATION_FIELDS = ['attempt_id', 'epoch', 'val_loss', 'val_macro_f1', 'val_balanced_accuracy', 'val_precision_per_class', 'val_recall_per_class', 'val_f1_per_class', 'is_best_checkpoint']
EPOCH_FIELDS = LEGACY_EPOCH_FIELDS.copy()
EPOCH_FIELDS.insert(EPOCH_FIELDS.index('train_macro_f1'), 'train_accuracy')
EPOCH_FIELDS.insert(EPOCH_FIELDS.index('val_macro_f1'), 'val_accuracy')
VALIDATION_FIELDS = LEGACY_VALIDATION_FIELDS.copy()
VALIDATION_FIELDS.insert(VALIDATION_FIELDS.index('val_macro_f1'), 'val_accuracy')


def append_epoch(path, row, fields, legacy):
    """Append idempotently; preserve old headers and add a v2 sidecar on legacy resume."""
    header = fields
    records = []
    if path.exists() and path.stat().st_size:
        with path.open(encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            header, records = reader.fieldnames, list(reader)
        if header not in (fields, legacy):
            raise ValueError(f'Log schema mismatch: {path}')
    key = (str(row.get('attempt_id', '')), str(row['epoch']))
    if not any((r.get('attempt_id', ''), r['epoch']) == key for r in records):
        append_csv(path, {k: row.get(k) for k in header}, header)
    if header == legacy and legacy != fields:
        append_epoch(path.with_name(path.stem + '.v2.csv'), row, fields, fields)


def append_csv(path, row, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(row)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding='utf-8', newline='') as stream:
            if next(csv.reader(stream)) != fields:
                raise ValueError(f'Log schema mismatch: {path}')
    with path.open('a', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
        stream.flush()


class RunLogger:
    def __init__(self, folder, echo=True):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        for name in ['training.log', 'errors.log', 'events.jsonl']:
            (self.folder / name).touch(exist_ok=True)

    def event(self, event, **details):
        row = {'timestamp': now(), 'event': event, **details}
        with (self.folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        message = f'[{event.upper()}] {details}'
        with (self.folder / 'training.log').open('a', encoding='utf-8') as stream:
            stream.write(now() + ' ' + message + '\n')
            stream.flush()
        if self.echo:
            print(message, flush=True)

    def message(self, text):
        """Plain text only; progress bars write directly to the terminal, never here."""
        with (self.folder / 'training.log').open('a', encoding='utf-8') as stream:
            stream.write(text + '\n')
            stream.flush()

    def iteration(self, row):
        append_csv(self.folder / 'iterations.csv', row, ITERATION_FIELDS)

    def epoch(self, row):
        append_epoch(self.folder / 'epochs.csv', row, EPOCH_FIELDS, LEGACY_EPOCH_FIELDS)

    def validation(self, row):
        append_epoch(self.folder / 'validation.csv', row, VALIDATION_FIELDS, LEGACY_VALIDATION_FIELDS)

    def error(self, error):
        with (self.folder / 'errors.log').open('a', encoding='utf-8') as stream:
            stream.write(now() + '\n' + ''.join(traceback.format_exception(error)) + '\n')
            stream.flush()
        self.event('failure', error=repr(error))
