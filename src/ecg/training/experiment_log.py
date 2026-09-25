"""Keep a log of every training run in results/experiments.csv, for report section 4.4."""
from pathlib import Path

import pandas as pd


def log_run(path: str | Path, row: dict) -> None:
    """Append one row to the CSV. Write the header only when the file is new."""
    path = Path(path)
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
