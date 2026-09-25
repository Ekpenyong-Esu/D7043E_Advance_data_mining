"""Load a YAML config file.

The training and noise configs have a line ``data: configs/preprocess.yaml``.
This module replaces that line with the contents of the other file, so the
preprocessing settings are written once and every script sees the same values.
"""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict.

    If the file has a ``data`` key that is a path to another YAML file,
    load that file too and put its contents under ``data``.
    """
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    data_ref = cfg.get("data")
    if isinstance(data_ref, str):  # a path, not an already-loaded dict
        cfg["data"] = load_config(data_ref)
    return cfg
