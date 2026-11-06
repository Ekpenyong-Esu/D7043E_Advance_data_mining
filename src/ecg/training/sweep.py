"""The model-selection sweep of report section 4.4, driven by configs/sweep.yaml.

The stages run in order and each one starts from the settings of the best run so far:
stage 1 compares ways to handle the class imbalance, stage 2 varies architecture,
regularisation, augmentation and the beat window, stage 3 varies the rest again around
the stage-2 winner. Every run is one row in results/experiments.csv; runs already in the
log are skipped, so the sweep can be resumed. Selection uses validation macro-F1 only.
"""
import json
from pathlib import Path

import pandas as pd

from ecg.config import deep_update, load_model_config
from ecg.data.pipeline import prepare_all_splits
from ecg.deployment.toolchain import import_ai8x
from ecg.logging_utils import get_logger
from ecg.training.experiment_log import has_run
from ecg.training.runner import run_cnn_training

log = get_logger(__name__)


def ensure_processed(cfg: dict) -> None:
    """Build the processed arrays named by the config if they do not exist yet (the wide-window variant)."""
    processed = Path(cfg["paths"]["processed"])
    if not all((processed / f"{s}.npz").exists() for s in ("train", "val", "test")):
        log.info("preparing %s", processed)
        prepare_all_splits(cfg["preprocess"])


def run_stage(runs: list[dict], base_overrides: dict, model_yaml: Path, root: Path, history_dir: Path) -> list[str]:
    """Train every run of one stage (on top of ``base_overrides``). Returns the run names."""
    names = []
    for run in runs:
        overrides = deep_update(base_overrides, run)
        cfg = load_model_config(model_yaml, root, overrides)
        name = f"{cfg['name']}_fp32"
        names.append(name)
        if has_run(cfg["paths"]["experiments"], name):
            log.info("%s already in the log, skipped", name)
            continue
        ensure_processed(cfg)
        result = run_cnn_training(cfg, import_ai8x(cfg))
        history_dir.mkdir(parents=True, exist_ok=True)
        result["history"].to_csv(history_dir / f"{name}.csv", index=False)
    return names


def best_row(experiments_csv: str | Path, names: list[str]) -> pd.Series:
    """The row with the highest validation macro-F1 among ``names``."""
    df = pd.read_csv(experiments_csv)
    return df[df["name"].isin(names)].sort_values("val_macro_f1", ascending=False).iloc[0]


def overrides_of(row: pd.Series) -> dict:
    """The config overrides that produced a logged run, without its name."""
    return {k: v for k, v in json.loads(row["overrides"]).items() if k != "name"}


def run_sweep(sweep: dict, model_yaml: str | Path, root: str | Path, history_dir: str | Path) -> pd.DataFrame:
    """Run every stage in order, each on top of the best run so far.

    Returns every sweep row of the log, best first, with a ``stage`` column.
    """
    model_yaml, root, history_dir = Path(model_yaml), Path(root), Path(history_dir)
    experiments = load_model_config(model_yaml, root)["paths"]["experiments"]
    stage_of: dict[str, int] = {}
    base: dict = {}
    for i, stage in enumerate(sweep["stages"], start=1):
        log.info("stage %d (%s) starts from %s", i, stage["name"], base or "model.yaml")
        for name in run_stage(stage["runs"], base, model_yaml, root, history_dir):
            stage_of[name] = i
        base = overrides_of(best_row(experiments, list(stage_of)))
    df = pd.read_csv(experiments)
    df = df[df["name"].isin(stage_of)].copy()
    df["stage"] = df["name"].map(stage_of)
    return df.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)
