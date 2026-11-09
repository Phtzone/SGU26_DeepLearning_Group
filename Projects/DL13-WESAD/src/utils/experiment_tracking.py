"""One metric stream to local files, console and optional local TensorBoard."""
from pathlib import Path
from src.utils.console import ConsoleLogger
from src.utils.logging import RunLogger
from src.utils.tensorboard_tracker import TensorBoardTracker
from src.utils.tracking import write_json


class TrainingTracker:
    def __init__(self, folder, config):
        self.folder = Path(folder)
        self.local = RunLogger(self.folder, echo=False)
        self.console = ConsoleLogger(config.get('logging', {}).get('console'))
        self.tensorboard = TensorBoardTracker(config, self.folder.parent / 'tensorboard', warning=self.warning)

    def warning(self, message):
        self.local.event('tracking_warning', message=message)
        self.console.write('WARNING: ' + message)

    def event(self, event, **details):
        self.local.event(event, **details)
        if event in ('resume', 'early_stopping', 'failure', 'tracking_warning'):
            self.console.write(f'[{event.upper()}] {details}')

    def error(self, error):
        self.local.error(error)
        self.console.write(f'[FAILURE] {type(error).__name__}: {error}')

    def epoch(self, row):
        self.local.epoch(row)

    def validation(self, row):
        self.local.validation(row)

    def start_run(self, metadata, group_id, total_folds, resume_state=None):
        self.console.header(metadata, total_folds)
        self.local.message('RUN HEADER ' + str({k: metadata[k] for k in ('run_id', 'fold_id', 'training_seed', 'window_counts', 'trainable_parameters', 'device')}))
        identity = self.tensorboard.start_run(metadata, group_id, resume_state)
        if identity:
            write_json(self.folder.parent / 'tracking_identity.json', identity)

    def start_epoch(self, epoch, total, batches):
        self.console.start_epoch(epoch, total, batches)

    def log_batch(self, row):
        self.local.iteration(row)
        self.console.batch(row)
        self.tensorboard.log_batch(row)

    def log_epoch(self, row, validation, context):
        self.epoch(row)
        self.validation(validation)
        self.console.epoch_summary(row, context)
        self.local.message(f"EPOCH SUMMARY epoch={row['epoch']} train_accuracy={row.get('train_accuracy', 'N/A')} val_accuracy={row.get('val_accuracy', 'N/A')} train_eval_loss={row['train_eval_loss']} val_loss={row['val_loss']} train_macro_f1={row['train_macro_f1']} val_macro_f1={row['val_macro_f1']} train_balanced_accuracy={row['train_balanced_accuracy']} val_balanced_accuracy={row['val_balanced_accuracy']}")
        self.tensorboard.log_epoch(row)
        self.tensorboard.log_validation(validation)
        self.tensorboard.flush()

    def log_checkpoint(self, path, epoch, score):
        self.tensorboard.log_checkpoint(path, epoch, score)

    def log_test(self, metrics, metadata, figures):
        self.console.fold_summary(metadata)
        self.tensorboard.log_test(metrics, figures, metadata)
        self.tensorboard.flush()

    def finish_run(self, status):
        self.console.close_bar()
        self.tensorboard.finish_run(status)
