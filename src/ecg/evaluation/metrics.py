"""Accuracy, macro-F1 and per-class precision / recall / F1. Pure functions."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ecg.constants import CLASS_NAMES


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASS_NAMES)))),
    }


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(len(CLASS_NAMES)), zero_division=0)
    return [
        {"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, c in enumerate(CLASS_NAMES)
    ]
