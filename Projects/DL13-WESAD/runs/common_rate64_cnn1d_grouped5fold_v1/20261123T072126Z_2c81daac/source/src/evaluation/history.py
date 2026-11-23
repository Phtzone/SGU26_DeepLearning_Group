"""Read-only, schema-aware committed history shared by import and plotting."""
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
import numpy as np
from src.training.checkpoint import load_checkpoint
from src.evaluation.metrics import classification_metrics
from src.utils.config import ROOT
from src.utils.tracking import read_json, file_hash

INTEGER_FIELDS = {'epoch', 'fold_id', 'training_seed', 'best_epoch', 'batch_index', 'global_step', 'batch_size', 'true_label', 'predicted_label', 'label', 'original_label', 'windows'}
TEXT_FIELDS = {'run_id', 'attempt_id', 'timestamp', 'subject_id', 'window_id'}


def parse_value(key, value):
    if value is None or value in ('', 'N/A', 'None', 'null'):
        return None
    if key in TEXT_FIELDS:
        return value
    if key == 'is_best_checkpoint':
        return str(value).lower() == 'true'
    if str(value).startswith(('[', '{')):
        return json.loads(value)
    if key in INTEGER_FIELDS:
        return int(value)
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except ValueError:
        return value


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return [{k: parse_value(k, v) for k, v in row.items()} for row in csv.DictReader(stream)]


def epoch_key(row):
    return str(row.get('attempt_id', '')), int(row['epoch'])


def merged_epochs(path, committed):
    merged = {}
    for candidate in (path, path.with_name(path.stem + '.v2.csv')):
        if not candidate.exists():
            continue
        seen = set()
        for row in read_csv(candidate):
            key = epoch_key(row)
            if key in seen:
                raise ValueError(f'Duplicate epoch record: {candidate}, {key}')
            seen.add(key)
            if key in committed:
                merged.setdefault(key, {}).update({k: v for k, v in row.items() if v is not None})
    return [merged[key] for key in sorted(committed, key=lambda key: key[1]) if key in merged]


@dataclass
class FoldHistory:
    folder: Path
    metadata: dict
    epochs: list
    validation: list
    iterations: list
    test: dict
    predictions: list
    orphan_iterations: int


def resolve_group(group_id, root=ROOT):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', group_id):
        raise ValueError('Invalid group ID.')
    matches = list((Path(root) / 'runs').glob('*/' + group_id))
    if len(matches) != 1:
        raise ValueError(f'Expected exactly one experiment group, found {len(matches)} for {group_id}.')
    return matches[0]


def read_group(group_id, root=ROOT):
    root = Path(root)
    group = resolve_group(group_id, root)
    manifest = read_json(group / 'experiment_manifest.json')
    folds, sources, all_windows = [], set([group / 'experiment_manifest.json']), set()
    for folder in sorted(group.glob('seed_*/fold_*')):
        metadata = read_json(folder / 'metadata.json')
        if metadata['status'] != 'completed':
            continue
        marker = folder / 'completion.json'
        if marker.exists():
            for relative, checksum in read_json(marker)['files'].items():
                if file_hash(folder / relative) != checksum:
                    raise ValueError(f'Historical artifact checksum mismatch: {folder / relative}')
        checkpoint_path = folder / 'checkpoints/last.pt'
        state = load_checkpoint(checkpoint_path)
        committed = {epoch_key(r) for r in state['history']}
        epochs = merged_epochs(folder / 'logs/epochs.csv', committed)
        validation = merged_epochs(folder / 'logs/validation.csv', committed)
        if len(epochs) != metadata['completed_epochs'] or len(validation) != len(epochs):
            raise ValueError('Missing committed epoch/validation logs.')
        batch_rows = read_csv(folder / 'logs/iterations.csv')
        iterations = [row for row in batch_rows if epoch_key(row) in committed]
        keys = [(r['attempt_id'], r['epoch'], r['batch_index']) for r in iterations]
        if len(set(keys)) != len(keys):
            raise ValueError('Duplicate committed training batches.')
        predictions = read_csv(folder / 'metrics/predictions.csv')
        for row in predictions:
            key = (metadata['training_seed'], row['subject_id'], row['window_id'])
            if key in all_windows:
                raise ValueError('Duplicate OOF window across folds.')
            all_windows.add(key)
        test = read_json(folder / 'metrics/test.json')
        recomputed = classification_metrics([r['true_label'] for r in predictions], [r['predicted_label'] for r in predictions])
        for key in ('accuracy', 'macro_f1', 'balanced_accuracy'):
            if key in test and not np.isclose(test[key], recomputed[key], atol=1e-12, rtol=0):
                raise ValueError('Historical predictions and metrics disagree.')
        # Test accuracy can be derived from complete predictions; epoch accuracy cannot.
        test.setdefault('accuracy', recomputed['accuracy'])
        folds.append(FoldHistory(folder, metadata, epochs, validation, iterations, test, predictions, len(batch_rows) - len(iterations)))
        sources.update([checkpoint_path, folder / 'metadata.json', folder / 'metrics/test.json', folder / 'metrics/predictions.csv'])
        sources.update((folder / 'logs').glob('*.csv'))
    if not folds:
        raise ValueError('No completed folds with usable history.')
    result = root / 'results' / group.parent.name / group.name
    source_hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(sources)}
    return {'root': root, 'group': group, 'result': result, 'manifest': manifest, 'folds': folds, 'source_hashes': source_hashes}
