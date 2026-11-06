"""Pick beats the model got wrong, for the error analysis in report section 5.5."""
import numpy as np
import pandas as pd

from ecg.constants import CLASS_NAMES, CLASS_TO_INDEX


def select_examples(y_true: np.ndarray, y_pred: np.ndarray, pairs: list[tuple[str, str]], k: int,
                    seed: int = 0) -> dict[tuple[str, str], np.ndarray]:
    """For each ``(true, predicted)`` pair such as ``("N", "S")``, return up to ``k`` random beat indices."""
    rng = np.random.default_rng(seed)
    out = {}
    for t, p in pairs:
        idx = np.flatnonzero((y_true == CLASS_TO_INDEX[t]) & (y_pred == CLASS_TO_INDEX[p]))
        out[(t, p)] = rng.choice(idx, size=min(k, len(idx)), replace=False) if len(idx) else idx
    return out


def error_counts(y_true: np.ndarray, y_pred: np.ndarray, pairs: list[tuple[str, str]]) -> dict[str, int]:
    """How many beats fall into each ``(true, predicted)`` pair, as ``"N->S": count``."""
    return {f"{t}->{p}": int(((y_true == CLASS_TO_INDEX[t]) & (y_pred == CLASS_TO_INDEX[p])).sum()) for t, p in pairs}


def per_record_table(record_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """One row per record: beats, accuracy, and the true and predicted count of every class."""
    rows = []
    for rid in np.unique(record_ids):
        m = record_ids == rid
        row = {"record": int(rid), "beats": int(m.sum()), "accuracy": float((y_true[m] == y_pred[m]).mean())}
        for i, c in enumerate(CLASS_NAMES):
            row[f"true {c}"] = int((y_true[m] == i).sum())
            row[f"pred {c}"] = int((y_pred[m] == i).sum())
        rows.append(row)
    return pd.DataFrame(rows)
