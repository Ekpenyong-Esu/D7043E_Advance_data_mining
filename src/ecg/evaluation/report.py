"""Append a model's metrics to results/metrics.csv (report §5.1 / §5.2)."""
from pathlib import Path


def append_metrics(path: str | Path, model_name: str, summary: dict, per_class: list[dict], n_params: int) -> None:
    raise NotImplementedError
