"""Validate fold provenance, audit committed logs and aggregate OOF predictions."""
import csv
import math
import re
from pathlib import Path
import numpy as np
from src.data.splits import load_manifest
from src.evaluation.metrics import classification_metrics, LABEL_NAMES
from src.evaluation.evaluator import write_csv
from src.evaluation.visualization import bar_plot, confusion_plot
from src.training.checkpoint import load_checkpoint
from src.training.trainer import completed_fold
from src.utils.config import ROOT, same_training_config
from src.utils.tracking import read_json, write_json, digest


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def validate_predictions(rows, fold, expected_rows):
    keys = [(r['subject_id'], r['window_id']) for r in rows]
    expected = {(r['subject_id'], r['window_id']): r for r in expected_rows}
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError('Duplicate, missing or extra OOF predictions.')
    if {r['subject_id'] for r in rows} != set(fold['test']):
        raise ValueError('OOF subjects differ from test manifest.')
    for row in rows:
        raw = expected[(row['subject_id'], row['window_id'])]
        if int(row['fold_id']) != fold['fold_id'] or int(row['true_label']) != int(raw['label']):
            raise ValueError('OOF fold or ground truth mismatch.')
        p = np.array([float(row['p_' + name]) for name in LABEL_NAMES])
        if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, atol=1e-5) or int(p.argmax()) != int(row['predicted_label']):
            raise ValueError('Invalid OOF probabilities/predictions.')
    return classification_metrics([int(r['true_label']) for r in rows], [int(r['predicted_label']) for r in rows])


def audit_iterations(folder, checkpoint, train_windows, batch_size):
    rows = read_csv(folder / 'logs/iterations.csv')
    committed = {(r['attempt_id'], r['epoch']) for r in checkpoint['history']}
    accepted = [r for r in rows if (r['attempt_id'], int(r['epoch'])) in committed]
    expected_batches = math.ceil(train_windows / batch_size)
    for attempt, epoch in committed:
        selected = [r for r in accepted if r['attempt_id'] == attempt and int(r['epoch']) == epoch]
        if sorted(int(r['batch_index']) for r in selected) != list(range(expected_batches)):
            raise ValueError('Missing/duplicate committed training iterations.')
        if sum(int(r['batch_size']) for r in selected) != train_windows:
            raise ValueError('Iteration sample coverage mismatch.')
    return {'logged_iterations': len(rows), 'committed_iterations': len(accepted), 'orphan_or_replayed_iterations': len(rows) - len(accepted), 'completed_epochs': len(committed)}


def stats(rows):
    return {metric: {'mean': float(np.mean([float(r[metric]) for r in rows])),
                     'std': float(np.std([float(r[metric]) for r in rows], ddof=1)) if len(rows) > 1 else None,
                     'n': len(rows)} for metric in ('macro_f1', 'balanced_accuracy', 'accuracy')}


def update_section(path, text):
    content = path.read_text(encoding='utf-8')
    begin, end = '<!-- ACTUAL_RESULTS_START -->', '<!-- ACTUAL_RESULTS_END -->'
    if begin not in content or end not in content:
        raise ValueError(f'Missing automatic results markers: {path}')
    path.write_text(content.split(begin)[0] + begin + '\n' + text + '\n' + end + content.split(end)[1], encoding='utf-8')


def aggregate_group(group, output=None, update_docs=True):
    manifest = read_json(group / 'experiment_manifest.json')
    config = manifest['effective_config']
    split = load_manifest(group / 'split_manifest.json')
    if digest(split) != manifest['manifest_checksum']:
        raise ValueError('Group split checksum mismatch.')
    output = output or ROOT / 'results/experiment_e1' / group.name
    output.mkdir(parents=True, exist_ok=True)
    folds, subjects, predictions, audits, histories = [], [], [], {}, []
    missing = []
    cache = Path(manifest['cache_path'])
    for seed in config['training']['seeds']:
        for fold in split['folds']:
            folder = group / f'seed_{seed}' / f'fold_{fold["fold_id"]:02d}'
            if not completed_fold(folder):
                missing.append({'training_seed': seed, 'fold_id': fold['fold_id']})
                continue
            meta = read_json(folder / 'metadata.json')
            if meta['manifest_checksum'] != manifest['manifest_checksum'] or not same_training_config(meta['effective_config'], config) or meta['subjects'] != fold or meta['training_seed'] != seed:
                raise ValueError('Completed fold does not belong to this experiment.')
            expected = [r for subject in fold['test'] for r in read_csv(cache / subject / 'windows.csv')]
            rows = read_csv(folder / 'metrics/predictions.csv')
            metrics = validate_predictions(rows, fold, expected)
            for metric in ('macro_f1', 'balanced_accuracy', 'accuracy'):
                if not np.isclose(metrics[metric], meta['test_metrics'][metric], atol=1e-12):
                    raise ValueError('Stored test metrics disagree with OOF predictions.')
            checkpoint = load_checkpoint(folder / 'checkpoints/last.pt')
            audits[f'{seed}/fold_{fold["fold_id"]:02d}'] = audit_iterations(folder, checkpoint, meta['window_counts']['train'], config['training']['batch_size'])
            histories.append((seed, fold['fold_id'], checkpoint['history']))
            folds.append({'fold_id': fold['fold_id'], 'training_seed': seed, 'run_id': meta['run_id'],
                          'train_subjects': ' '.join(fold['train']), 'validation_subjects': ' '.join(fold['validation']), 'test_subjects': ' '.join(fold['test']),
                          'best_epoch': meta['best_epoch'], 'completed_epochs': meta['completed_epochs'],
                          **{key: metrics[key] for key in ('macro_f1', 'balanced_accuracy', 'accuracy')},
                          'training_duration_seconds': meta['training_duration_seconds']})
            predictions.extend({'training_seed': seed, **r} for r in rows)
            for subject in fold['test']:
                sr = [r for r in rows if r['subject_id'] == subject]
                sm = classification_metrics([int(r['true_label']) for r in sr], [int(r['predicted_label']) for r in sr])
                subjects.append({'training_seed': seed, 'fold_id': fold['fold_id'], 'subject_id': subject, 'windows': len(sr),
                                 **{key: sm[key] for key in ('macro_f1', 'balanced_accuracy', 'accuracy')}})
    all_keys = [(r['training_seed'], r['subject_id'], r['window_id']) for r in predictions]
    if len(set(all_keys)) != len(all_keys):
        raise ValueError('Duplicate OOF predictions across folds.')
    status = 'partial' if missing else 'completed'
    summary = {'status': status, 'completed_folds': len(folds), 'expected_folds': 5 * len(config['training']['seeds']), 'missing_folds': missing,
               'group_id': group.name, 'manifest_checksum': manifest['manifest_checksum'], 'per_seed': {}}
    if not missing:
        for seed in config['training']['seeds']:
            fr, sr, pr = ([r for r in rows if r['training_seed'] == seed] for rows in (folds, subjects, predictions))
            if len(fr) != 5 or len(sr) != 15 or len({r['subject_id'] for r in sr}) != 15:
                raise ValueError('Incomplete five-fold / fifteen-subject coverage.')
            summary['per_seed'][str(seed)] = {'fold_level': stats(fr), 'subject_level': stats(sr),
                                              'pooled_oof': classification_metrics([int(r['true_label']) for r in pr], [int(r['predicted_label']) for r in pr]),
                                              'oof_windows': len(pr), 'training_duration_seconds': sum(r['training_duration_seconds'] for r in fr)}
    for filename, rows in [('fold_results.csv', folds), ('subject_metrics.csv', subjects), ('oof_predictions.csv', predictions)]:
        if rows:
            write_csv(output / filename, rows)
    write_json(output / 'experiment_manifest.json', manifest)
    write_json(output / 'iteration_audit.json', audits)
    write_json(output / 'summary.json', summary)
    text = f"Trạng thái: **{status}**. Hoàn tất {len(folds)}/{summary['expected_folds']} folds. Group: `{group.name}`.\n"
    if not missing:
        text += '\n| Seed | Fold | Epochs | Best epoch | Test Macro-F1 | Balanced Accuracy | Duration (s) |\n|---|---|---|---|---|---|---|\n'
        text += '\n'.join(f"| {r['training_seed']} | {r['fold_id']} | {r['completed_epochs']} | {r['best_epoch']} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} | {r['training_duration_seconds']:.1f} |" for r in folds)
        for seed, result in summary['per_seed'].items():
            text += f'\n\nSeed {seed}; sample standard deviation (`ddof=1`):\n\n| Aggregation | Macro-F1 | Balanced Accuracy |\n|---|---|---|\n'
            for key in ('fold_level', 'subject_level'):
                a, b = (result[key][m] for m in ('macro_f1', 'balanced_accuracy'))
                text += f"| {key} | {a['mean']:.4f} ± {a['std']:.4f} | {b['mean']:.4f} ± {b['std']:.4f} |\n"
            pooled = result['pooled_oof']
            text += f"| pooled_oof | {pooled['macro_f1']:.4f} | {pooled['balanced_accuracy']:.4f} |\n"
            seed_subjects = [r for r in subjects if str(r['training_seed']) == seed]
            seed_folds = [r for r in folds if str(r['training_seed']) == seed]
            lowest = min(seed_subjects, key=lambda r: r['macro_f1'])
            highest = max(seed_subjects, key=lambda r: r['macro_f1'])
            text += f"\nMacro-F1 giữa folds: {min(r['macro_f1'] for r in seed_folds):.4f}–{max(r['macro_f1'] for r in seed_folds):.4f}; giữa subjects: {lowest['subject_id']} ({lowest['macro_f1']:.4f}) đến {highest['subject_id']} ({highest['macro_f1']:.4f}). Pooled OOF có {result['oof_windows']} windows; tổng thời lượng folds {result['training_duration_seconds']:.1f} giây.\n"
            figures = output / 'figures' if len(summary['per_seed']) == 1 else output / f'figures/seed_{seed}'
            figures.mkdir(parents=True, exist_ok=True)
            fr = [r for r in folds if str(r['training_seed']) == seed]
            sr = [r for r in subjects if str(r['training_seed']) == seed]
            for metric in ('macro_f1', 'balanced_accuracy'):
                bar_plot([str(r['fold_id']) for r in fr], [r[metric] for r in fr], metric, figures / f'{metric}_by_fold.png')
            bar_plot([r['subject_id'] for r in sr], [r['macro_f1'] for r in sr], 'Macro-F1', figures / 'macro_f1_by_subject.png')
            confusion_plot(pooled['confusion_matrix'], figures / 'aggregate_confusion_matrix.png')
            np.savetxt(figures.parent / 'aggregate_confusion_matrix.csv', pooled['confusion_matrix'], fmt='%d', delimiter=',', header=','.join(LABEL_NAMES), comments='')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 4))
            for hs, fid, history in histories:
                if str(hs) == seed:
                    ax.plot([h['epoch'] for h in history], [h['val_macro_f1'] for h in history], label=f'fold {fid}')
            ax.set(xlabel='Epoch', ylabel='Validation Macro-F1')
            ax.legend()
            fig.tight_layout()
            fig.savefig(figures / 'training_curves.png', dpi=160)
            plt.close(fig)
        text += '\n\n| Seed | Subject | Fold | Windows | Macro-F1 | Balanced Accuracy |\n|---|---|---|---|---|---|\n'
        text += '\n'.join(f"| {r['training_seed']} | {r['subject_id']} | {r['fold_id']} | {r['windows']} | {r['macro_f1']:.4f} | {r['balanced_accuracy']:.4f} |" for r in subjects)
        text += '\n\nCác metrics được tính lại từ predictions và kiểm tra với manifest. Hiệu suất biến thiên theo subject trong WESAD; các kết quả này chưa xác định được nguyên nhân của khác biệt. Một training seed chưa đánh giá được độ ổn định theo khởi tạo. Chưa có kết quả E2/E3 để so sánh kiến trúc.'
    else:
        text += '\nChưa đủ folds; không báo cáo mean/std cuối cùng. Folds còn thiếu: ' + str(missing)
    (output / 'summary.md').write_text(text + '\n', encoding='utf-8')
    if update_docs:
        relative = '../../results/experiment_e1/' + group.name
        update_section(ROOT / 'experiments/e1_cnn/README.md', text + f'\n\n[Summary]({relative}/summary.md) · [OOF]({relative}/oof_predictions.csv) · [Figures]({relative}/figures/) · [Runs](../../runs/experiment_e1/{group.name}/)')
        update_section(ROOT / 'README.md', f"E1: **{status}**, {len(folds)}/{summary['expected_folds']} folds. [Kết quả](results/experiment_e1/{group.name}/summary.md) · [Chi tiết E1](experiments/e1_cnn/README.md).")
    return summary
