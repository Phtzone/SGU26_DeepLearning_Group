"""Fold-local weighted loss and Adam optimizer."""
import numpy as np
import torch


def make_optimizer(model, config):
    return torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])


def class_weights(labels):
    counts = np.bincount(labels, minlength=3)
    if (counts == 0).any():
        raise ValueError('Training subjects must contain all three classes.')
    return torch.tensor(len(labels) / (3 * counts), dtype=torch.float32)
