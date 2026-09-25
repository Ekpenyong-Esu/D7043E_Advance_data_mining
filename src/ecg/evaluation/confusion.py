"""Confusion matrix: how many beats of each true class were predicted as each class. Not written yet."""
from pathlib import Path

import numpy as np


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a 4 x 4 array. Rows are true classes, columns are predicted classes."""
    raise NotImplementedError


def save(cm: np.ndarray, out_stem: str | Path) -> None:
    """Write ``<stem>.csv`` and ``<stem>.png``."""
    raise NotImplementedError
