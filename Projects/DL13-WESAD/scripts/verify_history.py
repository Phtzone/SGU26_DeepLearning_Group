"""Technical regression audit; never aggregates into or rewrites historical results."""
import argparse
from pathlib import Path
from src.evaluation.history import read_group
from src.evaluation.metrics import classification_metrics
from src.utils.config import ROOT
from src.utils.tracking import file_hash, read_json, write_json


def verify(root=ROOT, baseline=None):
    root = Path(root)
    baseline = read_json(baseline or root / 'tests/fixtures/historical_e1_integrity.json')
    modified = [relative for relative, sha in baseline['files'].items() if not (root / relative).is_file() or file_hash(root / relative) != sha]
    for relative, block in baseline['readme_results_blocks'].items():
        current = (root / relative).read_text(encoding='utf-8').split('<!-- ACTUAL_RESULTS_START -->')[1].split('<!-- ACTUAL_RESULTS_END -->')[0]
        if current != block:
            modified.append(relative + ':actual_results_block')
    if modified:
        raise ValueError('Protected historical files changed: ' + ', '.join(modified))
    history = read_group(baseline['group_id'], root)
    folds = history['folds']
    predictions = [r for f in folds for r in f.predictions]
    pooled = classification_metrics([r['true_label'] for r in predictions], [r['predicted_label'] for r in predictions])
    counts = {'folds': len(folds), 'epochs': sum(len(f.epochs) for f in folds), 'iterations': sum(len(f.iterations) for f in folds), 'oof_windows': len(predictions)}
    if counts != {'folds': 5, 'epochs': 137, 'iterations': 5554, 'oof_windows': 2136}:
        raise ValueError('Historical E1 cardinality changed: ' + str(counts))
    expected = read_json(history['result'] / 'summary.json')['per_seed']['42']['pooled_oof']
    if pooled['macro_f1'] != expected['macro_f1'] or pooled['balanced_accuracy'] != expected['balanced_accuracy']:
        raise ValueError('Historical E1 test metrics changed.')
    return {'status': 'passed', 'protected_files': len(baseline['files']), **counts,
            'macro_f1': pooled['macro_f1'], 'balanced_accuracy': pooled['balanced_accuracy'],
            'split_manifest_sha256': file_hash(root / 'splits/grouped_5fold_seed42.json'), 'checkpoints_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/tracking_upgrade/regression_audit.json')
    args = parser.parse_args()
    result = verify()
    write_json(ROOT / args.output, result)
    print(result)


if __name__ == '__main__':
    main()
