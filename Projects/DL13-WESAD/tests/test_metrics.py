import numpy as np
import pytest
from src.evaluation.metrics import classification_metrics
from src.training.callbacks import SelectionState


def test_known_metrics():
    m = classification_metrics([0, 0, 1, 1, 2, 2], [0, 0, 1, 2, 2, 2])
    assert m['accuracy'] == pytest.approx(5 / 6)
    assert m['macro_f1'] == pytest.approx((1 + 2/3 + .8) / 3)
    assert m['balanced_accuracy'] == pytest.approx(5 / 6)
    assert np.sum(m['confusion_matrix']) == 6


def test_absent_class_policy():
    with pytest.warns(RuntimeWarning, match='Absent'):
        m = classification_metrics([0, 1], [0, 1])
    assert m['macro_f1'] == pytest.approx(2 / 3)
    assert m['balanced_accuracy'] == pytest.approx(2 / 3)


def test_validation_selection_and_patience():
    state = SelectionState(2, .001)
    assert state.update(.5, 1, 1) == (True, False)
    assert state.update(.5, .9, 2) == (True, False)
    assert state.best_epoch == 2
    assert state.update(.5005, .8, 3) == (True, True)
