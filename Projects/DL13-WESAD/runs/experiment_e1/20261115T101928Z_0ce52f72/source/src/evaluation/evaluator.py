"""Evaluation without gradients; weighted loss normalized over all samples."""
import csv
import numpy as np
import torch
from torch.nn import functional as F
from src.evaluation.metrics import classification_metrics, LABEL_NAMES
from src.utils.tracking import write_json


@torch.inference_mode()
def evaluate(model, loader, device, weights):
    model.eval()
    labels, predictions, probabilities, indices = [], [], [], []
    numerator, denominator = 0.0, 0.0
    for x, y, index in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('NaN/Inf evaluation logits')
        numerator += F.cross_entropy(logits, y, weight=weights, reduction='sum').item()
        denominator += weights[y].sum().item()
        probabilities.extend(logits.softmax(dim=1).cpu().numpy())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        labels.extend(y.cpu().tolist())
        indices.extend(index.tolist())
    metrics = classification_metrics(labels, predictions)
    metrics['loss'] = numerator / denominator
    return metrics, np.asarray(probabilities), np.asarray(indices)


def export_test(folder, dataset, metrics, probabilities, indices, fold_id):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for probability, index in zip(probabilities, indices):
        row = dataset.rows[index]
        rows.append({'fold_id': fold_id, 'subject_id': row['subject_id'], 'window_id': row['window_id'],
                     'start_time': row['start_time'], 'end_time': row['end_time'], 'true_label': row['label'],
                     'predicted_label': int(probability.argmax()),
                     **{f'p_{name}': float(probability[c]) for c, name in enumerate(LABEL_NAMES)}})
    write_csv(folder / 'predictions.csv', rows)
    subject_rows = []
    for subject in dataset.subjects:
        selected = [r for r in rows if r['subject_id'] == subject]
        m = classification_metrics([r['true_label'] for r in selected], [r['predicted_label'] for r in selected])
        subject_rows.append({'fold_id': fold_id, 'subject_id': subject, 'windows': len(selected), **{k: m[k] for k in ('macro_f1', 'balanced_accuracy', 'accuracy')}})
    write_csv(folder / 'subject_metrics.csv', subject_rows)
    write_json(folder / 'test.json', metrics)
    write_json(folder / 'classification_report.json', {name: {key: metrics[key + '_per_class'][i] for key in ['precision', 'recall', 'f1']} | {'support': metrics['support'][i]} for i, name in enumerate(LABEL_NAMES)})
    np.savetxt(folder / 'confusion_matrix.csv', metrics['confusion_matrix'], delimiter=',', fmt='%d', header=','.join(LABEL_NAMES), comments='')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot export empty results.')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
