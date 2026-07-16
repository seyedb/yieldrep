import numpy as np
import pytest

from yieldrep.evaluation.metrics import directional_accuracy, mae, rmse


def test_regression_metrics() -> None:
    y_true = np.array([1.0, -1.0, 2.0])
    y_pred = np.array([1.0, 0.0, -2.0])

    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(17.0 / 3.0))
    assert mae(y_true, y_pred) == pytest.approx(5.0 / 3.0)
    assert directional_accuracy(y_true, y_pred) == pytest.approx(1.0 / 3.0)
