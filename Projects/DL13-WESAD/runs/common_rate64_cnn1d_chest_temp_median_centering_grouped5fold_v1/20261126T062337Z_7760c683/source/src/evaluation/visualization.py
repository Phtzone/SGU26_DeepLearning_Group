"""Headless, exportable figures from actual history and predictions."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.evaluation.metrics import LABEL_NAMES


def confusion_plot(matrix, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, xlabel='Predicted', ylabel='True')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(matrix[i][j]), ha='center', va='center')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def training_plots(history, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for filename, fields in [('loss_curve.png', ['train_online_loss', 'train_eval_loss', 'val_loss']),
                             ('macro_f1_curve.png', ['train_macro_f1', 'val_macro_f1'])]:
        fig, ax = plt.subplots(figsize=(7, 4))
        for field in fields:
            ax.plot([r['epoch'] for r in history], [r[field] for r in history], label=field)
        ax.set_xlabel('Epoch')
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=160)
        plt.close(fig)


def bar_plot(labels, values, ylabel, path):
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * .5), 4))
    ax.bar(labels, values)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
