"""Keep a log of every training run in results/experiments.csv, for report section 4.4."""
from pathlib import Path

import pandas as pd


def has_run(path: str | Path, name: str) -> bool:
    """True when a run called ``name`` is already in the log, so the notebook can skip it."""
    path = Path(path)
    return path.exists() and bool((pd.read_csv(path)["name"] == name).any())


def log_run(path: str | Path, row: dict) -> pd.DataFrame:
    """Add one row. A row with the same ``name`` is replaced, so reruns do not pile up. Returns the table."""
    path = Path(path)
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df) and "name" in df:
        df = df[df["name"] != row["name"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
