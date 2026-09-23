"""Load YAML configs and resolve the nested ``data:`` reference."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML file. If it has a ``data`` key pointing at another YAML, inline it."""
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    data_ref = cfg.get("data")
    if isinstance(data_ref, str):
        cfg["data"] = load_config(data_ref)
    return cfg
