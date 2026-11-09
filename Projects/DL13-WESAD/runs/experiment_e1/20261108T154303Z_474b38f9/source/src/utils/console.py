"""TTY progress bars and plain, capture-friendly scientific summaries."""
import sys
from datetime import timedelta
from tqdm.auto import tqdm


def number(value, digits=4):
    return 'N/A' if value is None or value == '' else f'{float(value):.{digits}f}'


class ConsoleLogger:
    def __init__(self, options=None, stream=None):
        self.options = options or {}
        self.stream = stream or sys.stdout
        self.enabled = self.options.get('enabled', True)
        self.bar = None
        self.context = {}

    def write(self, text):
        if self.enabled:
            if self.bar:
                self.bar.write(text, file=self.stream)
            else:
                print(text, file=self.stream, flush=True)

    def header(self, metadata, total_folds):
        self.context = metadata
        self.total_folds = total_folds
        config = metadata['effective_config']
        tc = config['training']
        lines = ['=' * 68, 'DL13-WESAD | DEEP LEARNING EXPERIMENT', '=' * 68]
        values = {'Experiment': config['experiment']['id'], 'Model': config['experiment']['model'],
                  'Fold': f"{metadata['fold_id']:02d}/{total_folds:02d}", 'Training Seed': metadata['training_seed'],
                  'Split Seed': config['split']['split_seed'], 'Device': metadata['device']}
        values.update({role.title() + ' Subjects': len(metadata['subjects'][role]) for role in ('train', 'validation', 'test')})
        values.update({role.title() + ' Windows': count for role, count in metadata['window_counts'].items()})
        values.update({'Parameters': metadata['trainable_parameters'], 'Optimizer': tc['optimizer'],
                       'Learning Rate': tc['learning_rate'], 'Batch Size': tc['batch_size'],
                       'Max Epochs': tc['max_epochs'], 'Early Stopping': f"Patience {tc['early_stopping_patience']}"})
        lines.extend(f'{name:<22}: {value}' for name, value in values.items())
        self.write('\n'.join(lines + ['=' * 68]))

    def start_epoch(self, epoch, total, batches):
        self.close_bar()
        mode = self.options.get('progress', 'auto')
        interactive = mode is True or mode == 'always' or (mode == 'auto' and self.stream.isatty())
        if self.enabled and interactive:
            label = f"Fold {self.context['fold_id']:02d}/{self.total_folds:02d} | Epoch {epoch:03d}/{total} TRAIN"
            self.bar = tqdm(total=batches, desc=label, file=self.stream, leave=False, dynamic_ncols=True,
                            ascii=True, mininterval=.25, unit='batch')

    def batch(self, row):
        if self.bar:
            self.bar.set_postfix(loss=number(row['train_loss']), lr=number(row['learning_rate'], 6), step=row['global_step'], refresh=False)
            self.bar.update(1)

    def close_bar(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None

    def epoch_summary(self, row, context):
        self.close_bar()
        config = self.context['effective_config']
        elapsed = context.get('elapsed')
        elapsed_text = 'N/A' if elapsed is None else str(timedelta(seconds=int(elapsed)))
        lines = ['-' * 68, f"{config['experiment']['id']} | {config['experiment']['model']} | Fold {self.context['fold_id']:02d}/{self.total_folds:02d} | Epoch {row['epoch']:03d}/{config['training']['max_epochs']}", '-' * 68]
        for title, prefix in [('TRAIN (end-of-epoch evaluation)', 'train'), ('VALIDATION', 'val')]:
            lines.append(title)
            for label, key in [('Loss', 'eval_loss' if prefix == 'train' else 'loss'), ('Accuracy', 'accuracy'), ('Macro-F1', 'macro_f1'), ('Balanced Accuracy', 'balanced_accuracy')]:
                lines.append(f'  {label:<20}: {number(row.get(prefix + "_" + key))}')
        lines.extend(['OPTIMIZATION', f"  Train online loss   : {number(row.get('train_online_loss'))}",
                      f"  Learning Rate       : {number(row.get('learning_rate'), 6)}",
                      f"  Epoch Duration      : {number(row.get('epoch_duration_seconds'), 1)} s",
                      f"  Total Elapsed       : {elapsed_text}",
                      'MODEL SELECTION', f"  Best Epoch          : {row['best_epoch']}",
                      f"  Best Val Macro-F1   : {number(row['best_val_macro_f1'])}",
                      f"  Patience            : {context.get('bad_epochs', 'N/A')}/{config['training']['early_stopping_patience']}",
                      '  Checkpoint          : ' + ('New best saved' if context.get('is_best') else 'No improvement'), '-' * 68])
        self.write('\n'.join(lines))

    def fold_summary(self, metadata):
        self.close_bar()
        metrics = metadata['test_metrics']
        lines = ['=' * 68, f"FOLD {metadata['fold_id']:02d}/{self.total_folds:02d} COMPLETED", '=' * 68]
        values = {'Best Epoch': metadata['best_epoch'], 'Epochs Completed': metadata['completed_epochs'],
                  'Test Accuracy': number(metrics.get('accuracy')), 'Test Macro-F1': number(metrics.get('macro_f1')),
                  'Test Bal. Acc': number(metrics.get('balanced_accuracy')), 'Training Time (s)': number(metadata['training_duration_seconds'], 1),
                  'Checkpoint': metadata['checkpoints']['best']}
        self.write('\n'.join(lines + [f'{k:<22}: {v}' for k, v in values.items()] + ['=' * 68]))

    def cv_summary(self, metadata):
        self.write('Fold | Seed | Epochs | Best | Test Accuracy | Test Macro-F1 | Test Bal. Acc')
        for m in metadata:
            test = m['test_metrics']
            self.write(f"{m['fold_id']:4d} | {m['training_seed']:4d} | {m['completed_epochs']:6d} | {m['best_epoch']:4d} | {number(test.get('accuracy')):>13} | {number(test.get('macro_f1')):>13} | {number(test.get('balanced_accuracy')):>13}")
