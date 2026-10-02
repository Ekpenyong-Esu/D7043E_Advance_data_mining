"""Load a YAML config file."""
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)
