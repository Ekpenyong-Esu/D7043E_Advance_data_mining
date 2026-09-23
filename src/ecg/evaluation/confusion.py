"""Confusion matrix as CSV and figure."""
from pathlib import Path

import numpy as np


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def save(cm: np.ndarray, out_stem: str | Path) -> None:
    """Write ``<stem>.csv`` and ``<stem>.png``."""
    raise NotImplementedError
