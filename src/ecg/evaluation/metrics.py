"""Scores for a set of predictions: accuracy, macro-F1, and per-class precision, recall and F1.

Accuracy alone is misleading here, because about 90 percent of beats are N.
Macro-F1 gives every class the same weight, so it shows how the rare classes do.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from ecg.constants import CLASS_NAMES

LABELS = list(range(len(CLASS_NAMES)))


def summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return accuracy and macro-F1 as a small dict."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # ``labels`` lists all four classes, so a class that is never predicted still counts as 0.
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)),
    }


def per_class(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """One dict per class with precision, recall (sensitivity), F1 and support (number of true beats)."""
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    return [{"class": c, "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CLASS_NAMES)]


def flat_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Everything above in one flat dict: ``accuracy``, ``macro_f1``, ``precision_N``, ``recall_N``, ``f1_N``, ..."""
    out = summary(y_true, y_pred)
    for row in per_class(y_true, y_pred):
        for k in ("precision", "recall", "f1"):
            out[f"{k}_{row['class']}"] = row[k]
    return out
