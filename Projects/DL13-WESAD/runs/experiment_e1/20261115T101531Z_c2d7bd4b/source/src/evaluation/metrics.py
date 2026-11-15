"""Fixed label order, including explicit handling of absent classes."""
import warnings
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

LABEL_NAMES = ['baseline', 'stress', 'amusement']


def classification_metrics(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if not len(labels) or labels.shape != predictions.shape or not np.isin(labels, [0, 1, 2]).all() or not np.isin(predictions, [0, 1, 2]).all():
        raise ValueError('Metrics require nonempty aligned three-class labels.')
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1, 2], zero_division=0)
    absent = [LABEL_NAMES[i] for i in np.flatnonzero(support == 0)]
    if absent:
        warnings.warn('Absent classes: ' + ', '.join(absent) + '; fixed-label macro scores assign zero recall/F1.', RuntimeWarning)
    return {'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall.mean()),
            'accuracy': float((labels == predictions).mean()), 'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(), 'f1_per_class': f1.tolist(), 'support': support.tolist(),
            'confusion_matrix': confusion_matrix(labels, predictions, labels=[0, 1, 2]).tolist(),
            'label_order': LABEL_NAMES, 'absent_classes': absent}
