import json
import numpy as np
import pytest
from PIL import Image
from src.evaluation.enhanced_visualization import row_normalize, prediction_metrics, FigureWriter, generate_visualizations, plt
from src.utils.tracking import file_hash


def test_normalization_and_fixed_class_order():
    matrix = np.array([[2, 2, 0], [0, 0, 0], [0, 1, 3]])
    np.testing.assert_allclose(row_normalize(matrix), [[.5, .5, 0], [0, 0, 0], [0, .25, .75]])
    metrics = prediction_metrics([{'true_label': t, 'predicted_label': p} for t, p in zip([0, 0, 0, 1, 2], [0, 0, 0, 0, 0])])
    assert metrics['label_order'] == ['baseline', 'stress', 'amusement']
    assert metrics['accuracy'] == pytest.approx(.6)
    assert metrics['balanced_accuracy'] == pytest.approx(1/3)


def test_png_svg_quality_and_no_overwrite(tmp_path):
    writer = FigureWriter(tmp_path, 'TEST_MODEL', {})
    writer.confusion('matrix', 'Unit test', [[3, 1, 0], [0, 2, 1], [0, 0, 4]], True)
    with Image.open(tmp_path / 'matrix.png') as image:
        assert image.info['dpi'][0] >= 200
    svg = (tmp_path / 'matrix.svg').read_text()
    assert all(name in svg for name in ['Baseline', 'Stress', 'Amusement'])
    original = file_hash(tmp_path / 'matrix.png')
    with pytest.raises(FileExistsError):
        writer.confusion('matrix', 'Unit test', np.eye(3), True)
    assert file_hash(tmp_path / 'matrix.png') == original
    assert not plt.get_fignums()


def test_historical_visualizations_missing_metrics_and_reuse(synthetic_history, capsys):
    root, group, folder = synthetic_history
    original = file_hash(folder / 'plots/original.png')
    manifest, output = generate_visualizations('unit_group', root)
    assert '[WARNING]' in capsys.readouterr().out
    assert manifest['status'] == 'completed'
    assert manifest['figure_count'] > 10
    assert any(item['figure'].endswith('/accuracy') for item in manifest['skipped'])
    assert not list(output.glob('training/**/accuracy.png'))
    assert manifest['metrics_by_seed']['42']['subjects']['S8']['macro_f1'] == 0
    assert len(manifest['metrics_by_seed']['42']['folds']) == 1
    checksums = {str(p): file_hash(p) for p in output.rglob('*') if p.is_file()}
    same, repeated = generate_visualizations('unit_group', root)
    assert repeated == output and same == manifest
    assert checksums == {str(p): file_hash(p) for p in repeated.rglob('*') if p.is_file()}
    assert file_hash(folder / 'plots/original.png') == original
    learning = [entry for entry in manifest['images']
                if entry['figure'].startswith('cross_fold/') and '/learning_' in entry['figure']]
    assert learning and all('Train / Validation' in entry['title'] and 'OOF' not in entry['title'] for entry in learning)
    training_metadata = [entry for entry in manifest['images'] if entry['figure'].endswith(('/best_epoch', '/completed_epochs', '/training_duration_seconds'))]
    assert all('Training' in entry['title'] and 'OOF' not in entry['title'] for entry in training_metadata)
    assert not plt.get_fignums()


def test_label_version_reuses_only_unchanged_verified_figures(tmp_path):
    previous = tmp_path / 'v1'
    original = FigureWriter(previous, 'TEST', {})
    original.bars('unchanged', 'Test scores', ['fold 1'], [.5], 'Accuracy')
    original.bars('learning', 'Wrong OOF title', ['epoch 1'], [.4], 'Macro-F1')
    fingerprints = {p: file_hash(p) for p in previous.rglob('*') if p.is_file()}
    writer = FigureWriter(tmp_path / 'v2', 'TEST', {},
                          reuse={entry['figure']: entry for entry in original.images}, reuse_directory=previous)
    writer.bars('unchanged', 'Test scores', ['fold 1'], [.5], 'Accuracy')
    writer.bars('learning', 'Train / Validation', ['epoch 1'], [.4], 'Macro-F1')
    assert writer.reused_figures == 1
    assert writer.images[0] == original.images[0]
    assert 'Train / Validation' in (tmp_path / 'v2/learning.svg').read_text()
    assert fingerprints == {p: file_hash(p) for p in previous.rglob('*') if p.is_file()}
