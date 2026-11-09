"""Frozen rotating grouped subject splits; never random window splits."""
import random
from collections import Counter
from pathlib import Path
from src.utils.tracking import canonical, digest, read_json

SUBJECTS = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']


def create_manifest(seed=42):
    order = SUBJECTS.copy()
    random.Random(seed).shuffle(order)
    groups = [order[i:i + 3] for i in range(0, 15, 3)]
    folds = [{'fold_id': k + 1, 'test': groups[k], 'validation': groups[(k + 1) % 5],
              'train': [s for j, group in enumerate(groups) if j not in {k, (k + 1) % 5} for s in group]}
             for k in range(5)]
    return {'protocol': 'grouped_subject_wise_5fold', 'algorithm_version': 'python-random-rotate-v1',
            'split_seed': seed, 'original_subject_order': SUBJECTS, 'shuffled_subject_order': order,
            'groups': groups, 'folds': folds}


def validate_manifest(manifest):
    if manifest != create_manifest(manifest['split_seed']):
        raise ValueError('Manifest differs from the specified deterministic algorithm.')
    for role, count in [('train', 3), ('validation', 1), ('test', 1)]:
        counts = Counter(s for fold in manifest['folds'] for s in fold[role])
        if counts != Counter({s: count for s in SUBJECTS}):
            raise ValueError('Subject coverage failure.')
    for fold in manifest['folds']:
        sets = [set(fold[r]) for r in ('train', 'validation', 'test')]
        if [len(x) for x in sets] != [9, 3, 3] or len(set.union(*sets)) != 15:
            raise ValueError('Subject overlap or invalid fold size.')
    return manifest


def load_manifest(path):
    path = Path(path)
    manifest = read_json(path)
    if path.with_suffix('.sha256').read_text().strip() != digest(manifest):
        raise ValueError('Split manifest checksum mismatch.')
    return validate_manifest(manifest)


def ensure_manifest(path, seed=42):
    path = Path(path)
    if path.exists():
        manifest = load_manifest(path)
        if manifest['split_seed'] != seed:
            raise ValueError('Existing manifest uses another seed; never overwrite it.')
        return manifest
    if path.with_suffix('.sha256').exists():
        raise ValueError('Orphan checksum exists; refusing to overwrite.')
    manifest = validate_manifest(create_manifest(seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(manifest))
    path.with_suffix('.sha256').write_text(digest(manifest) + '\n', encoding='ascii')
    return manifest
