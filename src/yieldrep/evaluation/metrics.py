from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rmse(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def directional_accuracy(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))
