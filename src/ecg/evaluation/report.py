"""The rows of results/metrics.csv, the table behind report sections 5.1 to 5.3."""
from pathlib import Path

import numpy as np
import pandas as pd

from ecg.evaluation.metrics import flat_scores


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, n_params: int, **extra) -> dict:
    """One flat row: model name, parameter count, accuracy, macro-F1 and the per-class scores."""
    return {"model": name, "n_params": int(n_params), **extra, **flat_scores(y_true, y_pred)}


def upsert_row(path: str | Path, row: dict, key: str = "model") -> pd.DataFrame:
    """Add ``row`` to the CSV at ``path``. A row with the same ``key`` is replaced. Returns the table."""
    path = Path(path)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df) and key in df:
        df = df[df[key] != row[key]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
