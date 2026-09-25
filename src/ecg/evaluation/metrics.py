"""Scores for a set of predictions: accuracy, macro-F1, and per-class precision, recall and F1.

Accuracy alone is misleading here, because about 90 percent of beats are N.
Macro-F1 gives every class the same weight, so it shows how the rare classes do.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ecg.constants import CLASS_NAMES


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy and macro-F1 as a small dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # ``labels`` lists all four classes, so a class with no predictions still counts as 0.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASS_NAMES)))),
    }


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Return one dict per class with precision, recall, F1 and support (number of true beats)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    return [
        {"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(CLASS_NAMES)
    ]
