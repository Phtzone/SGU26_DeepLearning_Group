"""Atomic trusted local checkpoints. last.pt defines committed epochs."""
import os
import uuid
import torch
from pathlib import Path


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    # Only checkpoints produced by this project in the local run directory.
    return torch.load(path, map_location='cpu', weights_only=False)
