"""The noise-robustness experiment, report section 5.4.

Noise from the Noise Stress Test Database is added to the raw DS2 signals, then the
same preprocessing as in training is run, then the frozen models are scored. Nothing
here touches training. The noisy data is built on the fly and never saved.
"""
import numpy as np
import pandas as pd

from ecg.constants import DS2
from ecg.data import pipeline
from ecg.evaluation.metrics import flat_scores
from ecg.evaluation.predict import Predictor
from ecg.logging_utils import get_logger
from ecg.noise.nstdb import load_noise
from ecg.noise.snr import add_noise

log = get_logger(__name__)


def noisy_split(pre_cfg: dict, noise: np.ndarray | None, snr_db: float | None, seed: int,
                records: list[int] = DS2) -> dict[str, np.ndarray]:
    """DS2 arrays (X, y, rr, record_ids) after adding ``noise`` at ``snr_db`` to every raw record.

    ``noise=None`` gives the clean split, built by the same code path. The noise offset is
    random per record but fixed by ``seed``, so the experiment repeats exactly.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for rid in records:
        rec, x = pipeline.read_signal(pre_cfg, rid)
        if noise is not None:
            x = add_noise(x, noise, snr_db, rng)
        p = pipeline.beats_from_signal(pre_cfg, rec, x)
        p["record_ids"] = np.full(len(p["y"]), rid)
        parts.append(p)
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "y", "rr", "record_ids")}


def run_noise_experiment(cfg: dict, predictors: dict[str, Predictor], noise_types: list[str],
                         snr_levels: list[float], pre_cfg: dict | None = None) -> pd.DataFrame:
    """Score every predictor on clean DS2 and on every (noise type, SNR) condition.

    ``pre_cfg`` lets the caller pass a changed preprocessing config, for example with the
    low-pass filter disabled, to test whether the preprocessing helps under noise.
    Returns one row per (model, noise, snr) with accuracy, macro-F1 and the per-class scores.
    """
    pre_cfg = pre_cfg or cfg["preprocess"]
    noises = {k: load_noise(cfg["paths"]["raw_nstdb"], k) for k in noise_types}
    conditions = [("none", "clean", None)] + [(k, snr, noises[k]) for k in noise_types for snr in snr_levels]
    rows = []
    for kind, snr, noise in conditions:
        d = noisy_split(pre_cfg, noise, None if noise is None else float(snr), cfg["seed"])
        for name, predict in predictors.items():
            scores = flat_scores(d["y"], predict(d["X"], d["rr"]))
            rows.append({"model": name, "noise": kind, "snr_db": snr, **scores})
            log.info("%s noise=%s snr=%s macro_f1=%.3f acc=%.3f", name, kind, snr, scores["macro_f1"], scores["accuracy"])
    return pd.DataFrame(rows)
