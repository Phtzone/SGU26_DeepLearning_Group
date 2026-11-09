"""Scientific figures from immutable history, never from regenerated model outputs."""
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
from src.evaluation.visualization import plt
from src.evaluation.history import read_group, read_csv
from src.evaluation.metrics import classification_metrics, LABEL_NAMES
from src.utils.config import ROOT
from src.utils.tracking import digest, file_hash, read_json, write_json, now

CLASS_NAMES = [name.title() for name in LABEL_NAMES]
COLORS = ['#0072B2', '#D55E00', '#009E73']
STYLE_VERSION = 'scientific-v1'


def row_normalize(matrix):
    matrix = np.asarray(matrix, dtype=float)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals != 0)


def prediction_metrics(rows):
    return classification_metrics([r['true_label'] for r in rows], [r['predicted_label'] for r in rows])


class FigureWriter:
    def __init__(self, output, model, sources):
        self.output, self.model, self.sources = Path(output), model, sources
        self.images, self.skipped = [], []

    def save(self, fig, stem, title):
        path = self.output / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        files = []
        try:
            fig.suptitle(self.model + ' | ' + title, fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, .94))
            for suffix in ('.png', '.svg'):
                target = path.with_suffix(suffix)
                if target.exists():
                    raise FileExistsError(f'Refusing to overwrite figure: {target}')
                fig.savefig(target, dpi=220, bbox_inches='tight')
                files.append({'path': target.relative_to(self.output).as_posix(), 'sha256': file_hash(target)})
            self.images.append({'figure': str(stem), 'title': title, 'files': files})
        finally:
            plt.close(fig)

    def skip(self, figure, reason):
        self.skipped.append({'figure': figure, 'status': 'skipped', 'reason': reason})

    def line(self, stem, title, series, xlabel, ylabel, best=None, zero=False):
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for index, (label, xs, ys) in enumerate(series):
            ax.plot(xs, ys, label=label, color=COLORS[index % len(COLORS)], linewidth=1.4)
        if best is not None:
            ax.axvline(best, color='#666666', linestyle='--', linewidth=1, label=f'Best epoch {best}')
        if zero:
            ax.axhline(0, color='#777777', linewidth=.8)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
        self.save(fig, stem, title)

    def bars(self, stem, title, labels, values, ylabel, bounded=False):
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * .65), 4.8))
        ax.bar(labels, values, color=COLORS[0])
        ax.set_ylabel(ylabel)
        if bounded:
            ax.set_ylim(0, 1)
        if len(labels) > 8:
            ax.tick_params(axis='x', labelrotation=45)
        self.save(fig, stem, title)

    def grouped(self, stem, title, labels, groups, ylabel, bounded=True):
        fig, ax = plt.subplots(figsize=(max(7, len(labels) * .7), 4.8))
        xs = np.arange(len(labels))
        width = .8 / len(groups)
        for i, (name, values) in enumerate(groups.items()):
            ax.bar(xs - .4 + width * (i + .5), values, width, label=name, color=COLORS[i % len(COLORS)])
        ax.set(xticks=xs, xticklabels=labels, ylabel=ylabel)
        if bounded:
            ax.set_ylim(0, 1)
        if len(labels) > 8:
            ax.tick_params(axis='x', labelrotation=45)
        ax.legend(fontsize=8)
        self.save(fig, stem, title)

    def heatmap(self, stem, title, values, row_names, column_names, normalized=True):
        values = np.asarray(values)
        fig, ax = plt.subplots(figsize=(7, max(4.5, len(row_names) * .35)))
        plot = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None, aspect='auto')
        ax.set(xticks=range(len(column_names)), yticks=range(len(row_names)), xticklabels=column_names, yticklabels=row_names,
               xlabel='Class', ylabel='Subject / fold')
        for i in range(len(row_names)):
            for j in range(len(column_names)):
                ax.text(j, i, f'{values[i, j]:.2f}' if normalized else f'{values[i,j]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if normalized and values[i, j] > .65 else 'black')
        fig.colorbar(plot, ax=ax, label='F1 score' if normalized else 'Count')
        self.save(fig, stem, title)

    def confusion(self, stem, title, matrix, normalized=False):
        values = row_normalize(matrix) if normalized else np.asarray(matrix)
        fig, ax = plt.subplots(figsize=(6, 5))
        image = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        ax.set(xticks=range(len(CLASS_NAMES)), yticks=range(len(CLASS_NAMES)), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel='Predicted class', ylabel='True class')
        for i, row in enumerate(values):
            for j, value in enumerate(row):
                text = f'{value:.1%}' if normalized else f'{int(value)}'
                ax.text(j, i, text, ha='center', va='center', color='white' if value > values.max() * .6 else 'black')
        fig.colorbar(image, ax=ax, label='Recall within true class' if normalized else 'Windows (count)')
        self.save(fig, stem, title)


def values(rows, key):
    return [float(r[key]) if r.get(key) is not None else np.nan for r in rows]


def learning_figures(writer, fold):
    seed, fid = fold.metadata['training_seed'], fold.metadata['fold_id']
    prefix = f'training/seed_{seed}/fold_{fid:02d}'
    title = f'Seed {seed} | Fold {fid:02d}'
    epochs = [r['epoch'] for r in fold.epochs]
    best = fold.metadata['best_epoch']
    for name, fields, ylabel in [('loss', [('Train eval', 'train_eval_loss'), ('Validation', 'val_loss'), ('Train online', 'train_online_loss')], 'Weighted cross-entropy loss'),
                                  ('accuracy', [('Train', 'train_accuracy'), ('Validation', 'val_accuracy')], 'Accuracy'),
                                  ('macro_f1', [('Train', 'train_macro_f1'), ('Validation', 'val_macro_f1')], 'Macro-F1'),
                                  ('balanced_accuracy', [('Train', 'train_balanced_accuracy'), ('Validation', 'val_balanced_accuracy')], 'Balanced accuracy')]:
        if not all(any(r.get(key) is not None for r in fold.epochs) for _, key in fields):
            writer.skip(prefix + '/' + name, 'Required epoch metrics are absent in the historical schema; no estimates or interpolation were used.')
            continue
        writer.line(prefix + '/' + name, title + ' | Train vs validation ' + name.replace('_', ' '),
                    [(label, epochs, values(fold.epochs, key)) for label, key in fields], 'Epoch', ylabel, best=best)
    steps = [r['global_step'] for r in fold.iterations]
    for name, key, ylabel in [('batch_loss', 'train_loss', 'Batch weighted cross-entropy loss'),
                              ('gradient_norm', 'gradient_norm', 'Gradient L2 norm (before clipping)')]:
        if any(r.get(key) is not None for r in fold.iterations):
            writer.line(prefix + '/' + name, title + ' | ' + name.replace('_', ' '), [('Training', steps, values(fold.iterations, key))], 'Committed global step', ylabel)
        else:
            writer.skip(prefix + '/' + name, key + ' unavailable')
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(epochs, values(fold.epochs, 'learning_rate'), color=COLORS[0])
    axes[0].set(xlabel='Epoch', ylabel='Learning rate')
    axes[1].plot(steps, values(fold.iterations, 'learning_rate'), color=COLORS[0])
    axes[1].set(xlabel='Committed global step', ylabel='Learning rate')
    writer.save(fig, prefix + '/learning_rate', title + ' | Learning rate')
    gap = np.asarray(values(fold.epochs, 'train_macro_f1')) - np.asarray(values(fold.epochs, 'val_macro_f1'))
    writer.line(prefix + '/macro_f1_gap', title + ' | Train-validation gap', [('Train minus validation', epochs, gap)], 'Epoch', 'Macro-F1 difference', best=best, zero=True)
    writer.line(prefix + '/epoch_duration', title + ' | Epoch duration', [('Duration', epochs, values(fold.epochs, 'epoch_duration_seconds'))], 'Epoch', 'Duration (seconds)')


def evaluation_figures(writer, fold):
    seed, fid = fold.metadata['training_seed'], fold.metadata['fold_id']
    suffix = f'seed_{seed}/fold_{fid:02d}'
    title = f'Seed {seed} | Fold {fid:02d} | Test'
    metrics = prediction_metrics(fold.predictions)
    writer.confusion('confusion_matrices/' + suffix + '/raw', title + ' confusion matrix', metrics['confusion_matrix'])
    writer.confusion('confusion_matrices/' + suffix + '/normalized', title + ' row-normalized confusion matrix', metrics['confusion_matrix'], True)
    writer.grouped('class_metrics/' + suffix + '/precision_recall_f1', title + ' | Per-class performance', CLASS_NAMES,
                   {name: metrics[name.lower() + '_per_class'] for name in ('Precision', 'Recall', 'F1')}, 'Score')
    true = np.bincount([r['true_label'] for r in fold.predictions], minlength=len(CLASS_NAMES))
    pred = np.bincount([r['predicted_label'] for r in fold.predictions], minlength=len(CLASS_NAMES))
    writer.grouped('class_metrics/' + suffix + '/label_distribution', title + ' | Label distribution', CLASS_NAMES, {'True': true, 'Predicted': pred}, 'Windows (count)', False)
    return metrics


def seed_figures(writer, seed, folds):
    rows = [r for f in folds for r in f.predictions]
    title = f'Seed {seed} | OOF test'
    metrics_by_subject = {}
    for subject in sorted({r['subject_id'] for r in rows}, key=lambda s: int(s.lstrip('S')) if s.lstrip('S').isdigit() else s):
        selected = [r for r in rows if r['subject_id'] == subject]
        metrics = prediction_metrics(selected)
        metrics_by_subject[subject] = metrics
        target = f'error_analysis/seed_{seed}/S8' if subject == 'S8' else f'subject_metrics/seed_{seed}/{subject}'
        writer.confusion(target + '/confusion_raw', title + f' | {subject} confusion matrix', metrics['confusion_matrix'])
        writer.confusion(target + '/confusion_normalized', title + f' | {subject} row-normalized confusion matrix', metrics['confusion_matrix'], True)
    subjects = list(metrics_by_subject)
    fold_labels = [str(f.metadata['fold_id']) for f in folds]
    for key, label in [('macro_f1', 'Macro-F1'), ('balanced_accuracy', 'Balanced accuracy'), ('accuracy', 'Accuracy')]:
        writer.bars(f'subject_metrics/seed_{seed}/{key}', title + ' | ' + label + ' by subject', subjects, [metrics_by_subject[s][key] for s in subjects], label, True)
        writer.bars(f'cross_fold/seed_{seed}/{key}', title + ' | ' + label + ' by fold', fold_labels, [f.test[key] for f in folds], label, True)
    writer.heatmap(f'subject_metrics/seed_{seed}/class_f1_heatmap', title + ' | Subject x class F1', [metrics_by_subject[s]['f1_per_class'] for s in subjects], subjects, CLASS_NAMES)
    writer.grouped(f'subject_metrics/seed_{seed}/class_f1', title + ' | Per-class F1 by subject', subjects,
                   {name: [metrics_by_subject[s]['f1_per_class'][c] for s in subjects] for c, name in enumerate(CLASS_NAMES)}, 'F1 score')
    writer.heatmap(f'cross_fold/seed_{seed}/class_f1_heatmap', title + ' | Fold x class F1', [f.test['f1_per_class'] for f in folds], fold_labels, CLASS_NAMES)
    writer.grouped(f'cross_fold/seed_{seed}/class_f1', title + ' | Per-class F1 by fold', fold_labels,
                   {name: [f.test['f1_per_class'][c] for f in folds] for c, name in enumerate(CLASS_NAMES)}, 'F1 score')
    fig, ax = plt.subplots(figsize=(7, 4.6))
    keys = ['macro_f1', 'balanced_accuracy', 'accuracy']
    ax.boxplot([[f.test[key] for f in folds] for key in keys], positions=range(len(keys)))
    ax.set(xticks=range(len(keys)), xticklabels=['Macro-F1', 'Balanced accuracy', 'Accuracy'], ylabel='Fold-level test score', ylim=(0, 1))
    writer.save(fig, f'cross_fold/seed_{seed}/metrics_boxplot', title + f' | Distribution across {len(folds)} folds')
    for key, label in [('best_epoch', 'Best epoch'), ('completed_epochs', 'Completed epochs'), ('training_duration_seconds', 'Training duration (seconds)')]:
        writer.bars(f'cross_fold/seed_{seed}/{key}', title + ' | ' + label + ' by fold', fold_labels, [f.metadata[key] for f in folds], label)
    pooled = prediction_metrics(rows)
    for normalized in (False, True):
        writer.confusion(f'cross_fold/seed_{seed}/aggregate_confusion_' + ('normalized' if normalized else 'raw'), title + ' | Pooled confusion matrix', pooled['confusion_matrix'], normalized)
    for key, train_key, val_key, ylabel in [('loss', 'train_eval_loss', 'val_loss', 'Weighted CE loss'), ('macro_f1', 'train_macro_f1', 'val_macro_f1', 'Macro-F1'), ('balanced_accuracy', 'train_balanced_accuracy', 'val_balanced_accuracy', 'Balanced accuracy'), ('accuracy', 'train_accuracy', 'val_accuracy', 'Accuracy')]:
        available = [f for f in folds if any(r.get(train_key) is not None and r.get(val_key) is not None for r in f.epochs)]
        if not available:
            writer.skip(f'cross_fold/seed_{seed}/learning_{key}', 'Epoch ' + key + ' unavailable; historical metrics were not manufactured.')
            continue
        cols = min(2, len(available))
        fig, axes = plt.subplots(math.ceil(len(available) / cols), cols, figsize=(6 * cols, 3.6 * math.ceil(len(available) / cols)), squeeze=False)
        for axis, fold in zip(axes.flat, available):
            epochs = [r['epoch'] for r in fold.epochs]
            for name, field, color in [('Train eval', train_key, COLORS[0]), ('Validation', val_key, COLORS[1])]:
                axis.plot(epochs, values(fold.epochs, field), label=name, color=color)
            axis.axvline(fold.metadata['best_epoch'], linestyle='--', color='#666666', label='Best epoch')
            axis.set(title=f"Fold {fold.metadata['fold_id']:02d}", xlabel='Epoch', ylabel=ylabel)
            axis.legend(fontsize=8)
        for axis in list(axes.flat)[len(available):]:
            axis.set_visible(False)
        writer.save(fig, f'cross_fold/seed_{seed}/learning_{key}', title + ' | Learning curves across folds')
    error_figures(writer, seed, rows, metrics_by_subject)
    return {'folds': {str(f.metadata['fold_id']): f.test for f in folds}, 'subjects': metrics_by_subject, 'pooled_oof': pooled}


def error_figures(writer, seed, rows, subject_metrics):
    selected = [r for r in rows if r['subject_id'] == 'S8']
    if not selected:
        writer.skip(f'error_analysis/seed_{seed}/S8', 'S8 is absent from these test predictions.')
        return
    prefix = f'error_analysis/seed_{seed}/S8'
    title = f'Seed {seed} | S8 | OOF test'
    writer.bars(prefix + '/class_f1', title + ' | Per-class F1', CLASS_NAMES, subject_metrics['S8']['f1_per_class'], 'F1 score', True)
    true = np.bincount([r['true_label'] for r in selected], minlength=len(CLASS_NAMES))
    predicted = np.bincount([r['predicted_label'] for r in selected], minlength=len(CLASS_NAMES))
    writer.grouped(prefix + '/label_distribution', title + ' | True vs predicted labels', CLASS_NAMES, {'True': true, 'Predicted': predicted}, 'Windows (count)', False)
    probability_keys = ['p_' + name for name in LABEL_NAMES]
    if all(all(row.get(k) is not None for k in probability_keys) for row in selected):
        confidence = np.array([max(r[k] for k in probability_keys) for r in selected])
        correct = np.array([r['true_label'] == r['predicted_label'] for r in selected])
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for mask, label, color in [(correct, 'Correct', COLORS[0]), (~correct, 'Incorrect', COLORS[1])]:
            if mask.any():
                ax.hist(confidence[mask], bins=np.linspace(0, 1, 21), alpha=.65, label=label, color=color)
        ax.set(xlabel='Maximum predicted probability', ylabel='Windows (count)', xlim=(0, 1))
        ax.legend()
        writer.save(fig, prefix + '/confidence_distribution', title + ' | Prediction confidence')
    else:
        writer.skip(prefix + '/confidence_distribution', 'Per-class probabilities are missing.')
    if all(row.get('start_time') is not None for row in selected):
        selected = sorted(selected, key=lambda r: r['start_time'])
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
        for ax, key, name in [(axes[0], 'true_label', 'True'), (axes[1], 'predicted_label', 'Predicted')]:
            ax.scatter([r['start_time'] for r in selected], [r[key] for r in selected], c=[COLORS[r[key]] for r in selected], s=13)
            ax.set(yticks=range(len(CLASS_NAMES)), yticklabels=CLASS_NAMES, ylabel=name)
        axes[1].set_xlabel('Window start time in recording (seconds)')
        writer.save(fig, prefix + '/label_timeline', title + ' | True vs predicted timeline (gaps preserved)')
    else:
        writer.skip(prefix + '/label_timeline', 'Window timestamps are missing.')
    others = [m for name, m in subject_metrics.items() if name != 'S8']
    if others:
        writer.grouped(prefix + '/comparison', title + ' | S8 vs other subjects', ['Macro-F1', 'Balanced accuracy', 'Accuracy'],
                       {'S8': [subject_metrics['S8'][k] for k in ('macro_f1', 'balanced_accuracy', 'accuracy')],
                        'Other subjects (unweighted mean)': [np.mean([m[k] for m in others]) for k in ('macro_f1', 'balanced_accuracy', 'accuracy')]}, 'Subject-level score')


def dataset_figures(writer, history):
    unique = {}
    for fold in history['folds']:
        for row in fold.predictions:
            unique[(row['subject_id'], row['window_id'])] = row
    rows = list(unique.values())
    subjects = sorted({r['subject_id'] for r in rows}, key=lambda s: int(s.lstrip('S')) if s.lstrip('S').isdigit() else s)
    counts = {s: np.bincount([r['true_label'] for r in rows if r['subject_id'] == s], minlength=len(CLASS_NAMES)) for s in subjects}
    writer.bars('dataset/windows_by_subject', 'Dataset | Valid windows by subject', subjects, [sum(counts[s]) for s in subjects], 'Valid windows (count)')
    writer.bars('dataset/class_distribution', 'Dataset | Class distribution', CLASS_NAMES, np.sum(list(counts.values()), axis=0), 'Valid windows (count)')
    writer.grouped('dataset/class_distribution_by_subject', 'Dataset | Class distribution by subject', subjects,
                   {name: [counts[s][c] for s in subjects] for c, name in enumerate(CLASS_NAMES)}, 'Valid windows (count)', False)
    first_seed = history['folds'][0].metadata['training_seed']
    for fold in history['folds']:
        if fold.metadata['training_seed'] == first_seed and 'label_distribution' in fold.metadata:
            writer.grouped(f'dataset/split_distribution_fold_{fold.metadata["fold_id"]:02d}', f"Dataset | Fold {fold.metadata['fold_id']:02d} split class counts",
                           CLASS_NAMES, {role.title(): distribution for role, distribution in fold.metadata['label_distribution'].items()}, 'Windows (count)', False)
    cache = Path(history['manifest'].get('cache_path', ''))
    metadata = [read_json(cache / s / 'metadata.json') for s in subjects if (cache / s / 'metadata.json').exists()]
    audited = [m for m in metadata if 'counts' in m]
    if audited:
        names = [m['subject_id'] for m in audited]
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        axes[0].bar(names, [m['counts']['kept_region_anchored_windows'] for m in audited], color=COLORS[0])
        axes[0].set(ylabel='Kept region-anchored windows (count)')
        invalid = [m['counts']['global_grid_rejections']['invalid_region'] for m in audited]
        transitions = [m['counts']['global_grid_rejections']['transition'] for m in audited]
        axes[1].bar(names, invalid, label='Invalid labels', color=COLORS[1])
        axes[1].bar(names, transitions, bottom=invalid, label='Transitions', color=COLORS[2])
        axes[1].set(ylabel='Rejected global-grid windows (count)', xlabel='Subject (different window grids; counts must not be added)')
        axes[1].legend()
        writer.save(fig, 'dataset/preprocessing_window_counts', 'Dataset | Preprocessing counts and separate global-grid audit')
    else:
        writer.skip('dataset/preprocessing_window_counts', 'Preprocessing count metadata is unavailable.')
    subject = 'S8' if 'S8' in subjects else subjects[0]
    folder = cache / subject
    if not (folder / 'windows.npy').exists() or not (folder / 'windows.csv').exists():
        writer.skip('dataset/signals_and_segmentation', 'Processed signal cache is unavailable; raw data is not synthesized.')
        return
    windows = np.load(folder / 'windows.npy', mmap_mode='r')
    records = read_csv(folder / 'windows.csv')
    cfg = history['manifest']['effective_config']['data']
    sensors = cfg['sensors']
    channels = {'ACC': 3, 'BVP': 1, 'EDA': 1}
    if sum(channels.get(sensor, 0) for sensor in sensors) != windows.shape[1]:
        writer.skip('dataset/signals', 'Cache channel layout differs from the stored sensor configuration.')
    else:
        fig, axes = plt.subplots(len(sensors), 1, figsize=(10, 2.5 * len(sensors)), sharex=True, squeeze=False)
        offset = 0
        times = np.arange(windows.shape[2]) / cfg['sampling_rate']
        for ax, sensor in zip(axes.flat, sensors):
            for c in range(channels[sensor]):
                ax.plot(times, windows[0, offset + c], linewidth=.8, color=COLORS[c], label=f'{sensor} {c + 1}')
            ax.set_ylabel(sensor + '\n(native units)')
            ax.legend(fontsize=8)
            offset += channels[sensor]
        axes[-1, 0].set_xlabel('Time within cached window (seconds)')
        writer.save(fig, 'dataset/physiological_signals', f'Dataset | {subject} | Resampled, unnormalized wrist signals')
    selected = records[:4]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, row in enumerate(selected):
        ax.broken_barh([(row['start_time'], row['end_time'] - row['start_time'])], (index - .35, .7), facecolors=COLORS[row['label']], alpha=.7)
    ax.set(yticks=range(len(selected)), yticklabels=[f'Window {i + 1}' for i in range(len(selected))], xlabel='Time in original recording (seconds)', ylabel='Retained window')
    writer.save(fig, 'dataset/sliding_window_segmentation', f'Dataset | {subject} | Actual window intervals (overlap retained)')


def generate_visualizations(group_id, root=ROOT):
    history = read_group(group_id, root)
    cache = Path(history['manifest'].get('cache_path', ''))
    sources = dict(history['source_hashes'])
    if cache.is_dir():
        for path in sorted(cache.glob('*/*')):
            if path.name in ('metadata.json', 'windows.csv', 'windows.npy'):
                sources[str(path)] = file_hash(path)
    generation_id = STYLE_VERSION + '_' + digest(sources)[:16]
    base = history['result'] / 'enhanced_visualizations'
    output = base / generation_id
    marker = output / 'generation_manifest.json'
    if marker.exists():
        manifest = read_json(marker)
        if manifest['status'] == 'completed' and all(file_hash(output / item['path']) == item['sha256'] for image in manifest['images'] for item in image['files']):
            print(f'[REUSE] {len(manifest["images"])} existing figures; no files overwritten.')
            return manifest, output
    if output.exists():
        # Failed/incomplete generations remain intact for inspection.
        from src.utils.tracking import identifier
        output = base / (generation_id + '_' + identifier())
    output.mkdir(parents=True)
    config = history['manifest']['effective_config']
    writer = FigureWriter(output, config['experiment']['id'] + ' / ' + config['experiment']['model'], sources)
    manifest = {'status': 'running', 'group_id': group_id, 'style_version': STYLE_VERSION, 'generated_at': now(),
                'source_hashes': sources, 'images': writer.images, 'skipped': writer.skipped, 'metrics_by_seed': {}}
    style = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.grid': True, 'axes.axisbelow': True,
             'grid.alpha': .18, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'}
    try:
        with plt.rc_context(style):
            seeds = defaultdict(list)
            for fold in history['folds']:
                learning_figures(writer, fold)
                evaluation_figures(writer, fold)
                seeds[fold.metadata['training_seed']].append(fold)
            for seed, folds in seeds.items():
                manifest['metrics_by_seed'][str(seed)] = seed_figures(writer, seed, folds)
            dataset_figures(writer, history)
        manifest.update(status='completed', figure_count=len(writer.images), file_count=sum(len(image['files']) for image in writer.images), dpi=220)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output / 'generation_manifest.json', manifest)
    write_json(base / 'generation_manifest.json', {**manifest, 'generation_directory': output.name})
    if writer.skipped:
        print(f'[WARNING] {len(writer.skipped)} figures skipped for unavailable inputs; see generation_manifest.json.')
    for entry in writer.images:
        for item in entry['files']:
            print(output / item['path'])
    print(f'Generated {len(writer.images)} figures / {manifest["file_count"]} PNG+SVG files; skipped {len(writer.skipped)}.')
    return manifest, output
