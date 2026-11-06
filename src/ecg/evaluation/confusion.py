"""Confusion matrix: how many beats of each true class were predicted as each class."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from ecg.constants import CLASS_NAMES


def compute(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return a 4 x 4 array. Rows are true classes, columns are predicted classes."""
    return confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))


def as_frame(cm: np.ndarray) -> pd.DataFrame:
    """The matrix as a labelled table (index = true class, columns = predicted class)."""
    return pd.DataFrame(cm, index=pd.Index(CLASS_NAMES, name="true"), columns=pd.Index(CLASS_NAMES, name="predicted"))


def save_csv(cm: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    as_frame(cm).to_csv(path)
