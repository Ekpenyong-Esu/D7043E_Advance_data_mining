"""Append one row per run to results/experiments.csv (feeds report §4.4)."""
from pathlib import Path

import pandas as pd


def log_run(path: str | Path, row: dict) -> None:
    path = Path(path)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
