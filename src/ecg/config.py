"""Load the YAML configs. A model config also pulls in the preprocessing config it names."""
import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read one YAML file into a dict."""
    with open(path) as fh:
        return yaml.safe_load(fh)


def deep_update(base: dict, overrides: dict | None) -> dict:
    """Return a copy of ``base`` with the keys of ``overrides`` written over it, block by block."""
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_model_config(path: str | Path, root: str | Path, overrides: dict | None = None) -> dict[str, Any]:
    """Read a model config (``configs/model.yaml``) and attach what it needs from ``configs/preprocess.yaml``.

    Adds ``fs``, ``input_length`` (the beat window) and ``input_scale`` (the clip limit, which maps
    the stored beats to the chip range [-1, 1)). Every path in ``paths:`` and ``toolchain:`` is
    made absolute under ``root``, so the functions that use the config need no other context.
    """
    root = Path(root)
    cfg = deep_update(load_config(path), overrides)
    pre = load_config(root / cfg["data"])
    cfg["preprocess"] = pre
    cfg["fs"] = int(pre["fs"])
    cfg["input_length"] = int(pre["beat"]["samples_before"]) + int(pre["beat"]["samples_after"])
    cfg["input_scale"] = float(pre["normalize"]["clip"])
    cfg["overrides"] = overrides or {}  # kept so a run can be rebuilt from its row in results/experiments.csv
    cfg.setdefault("paths", {})
    for k, v in pre["paths"].items():
        cfg["paths"].setdefault(k, v)
    for block in (cfg.get("paths", {}), cfg.get("toolchain", {}), pre["paths"]):
        for k, v in block.items():
            block[k] = str(root / v)
    return cfg
