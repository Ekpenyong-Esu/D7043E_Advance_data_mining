"""Add one model's scores to results/metrics.csv, the table behind report sections 5.1 and 5.2. Not written yet."""
from pathlib import Path


def append_metrics(path: str | Path, model_name: str, summary: dict, per_class: list[dict], n_params: int) -> None:
    """Append one row with accuracy, macro-F1, per-class scores and the parameter count."""
    raise NotImplementedError
