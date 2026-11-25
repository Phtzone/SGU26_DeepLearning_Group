"""Render CNN/GRU Test evidence from existing CSVs; no model evaluation or training."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def render(result):
    root = Path(__file__).resolve().parents[1]
    sources = {
        'CNN': root/'results/preprocessing_window_zscore_cnn_full/20261124_full_seed42',
        'GRU': root/'results/preprocessing_window_zscore_gru_full/20261124_full_seed42',
    }
    dest = result/'figures'; dest.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, sharey=True)
    for ax, (model, source) in zip(axes, sources.items()):
        control = {r['subject_id']: float(r['macro_f1']) for r in read_csv(result/model/'control/subject_metrics.csv')}
        new = {r['subject_id']: float(r['macro_f1']) for r in read_csv(result/model/'window_zscore/subject_metrics.csv')}
        subjects = sorted(control, key=lambda s: int(s[1:])); x = np.arange(len(subjects))
        ax.bar(x-.18, [control[s] for s in subjects], .36, label='Current control')
        ax.bar(x+.18, [new[s] for s in subjects], .36, label='Window z-score')
        ax.set(title=model, ylabel='Test Macro-F1', ylim=(0, 1)); ax.legend(loc='upper right')
        ax.set_xticks(x, subjects)
        history = [read_csv(source/f'fold_{fid:02d}/window_zscore/epochs.csv') for fid in range(1, 6)]
        curves, axs = plt.subplots(2, 5, figsize=(18, 6))
        for fid, rows in enumerate(history):
            epochs = [int(r['epoch']) for r in rows]
            for tag, color, label in [('train_loss', 'C0', 'Train'), ('val_loss', 'C1', 'Validation')]:
                axs[0, fid].plot(epochs, [float(r[tag]) for r in rows], color=color, label=label)
            for tag, color, label in [('train_macro_f1', 'C0', 'Train pooled'), ('val_macro_f1', 'C1', 'Val pooled'),
                                       ('val_subject_macro_f1', 'C2', 'Val mean/person')]:
                axs[1, fid].plot(epochs, [float(r[tag]) for r in rows], color=color, label=label)
            axs[0, fid].set_title(f'Fold {fid+1}'); axs[1, fid].set(xlabel='Epoch', ylim=(0, 1))
        axs[0, 0].set_ylabel('Weighted loss'); axs[1, 0].set_ylabel('Macro-F1')
        axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
        curves.suptitle(f'{model} | Window z-score | Train/Validation only | Selected by Val mean/person')
        curves.tight_layout(); curves.savefig(dest/f'{model}_training_curves.png', dpi=150); plt.close(curves)
        predictions = read_csv(result/model/'window_zscore/oof_predictions.csv')
        matrix = np.zeros((3, 3), dtype=int)
        for r in predictions: matrix[int(r['true_label']), int(r['predicted_label'])] += 1
        normalized = matrix/matrix.sum(axis=1, keepdims=True)
        cm, a = plt.subplots(figsize=(6, 5)); a.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
        labels = ['Baseline', 'Stress', 'Amusement']
        a.set(xticks=range(3), yticks=range(3), xticklabels=labels, yticklabels=labels,
              xlabel='Predicted class', ylabel='True class', title=f'{model} | Test confusion | All 15 people')
        for i in range(3):
            for j in range(3):
                a.text(j, i, f'{matrix[i,j]}\n{normalized[i,j]:.1%}', ha='center', va='center',
                       color='white' if normalized[i,j]>.6 else 'black')
        cm.tight_layout(); cm.savefig(dest/f'{model}_test_confusion.png', dpi=160); plt.close(cm)
    axes[-1].set_xlabel('Test subject'); fig.suptitle('Paired Test comparison by person | Frozen CNN/GRU checkpoints')
    fig.tight_layout(); fig.savefig(dest/'test_by_subject.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--result', type=Path, required=True)
    render(p.parse_args().result)
