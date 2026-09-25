"""Scale each beat so that all beats have a similar size. This is the last step before the model.

Different people and different electrodes give very different signal heights.
Normalising each beat on its own removes that difference.
"""
import numpy as np


def zscore(beats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Give each beat a mean of 0 and a standard deviation of 1. ``eps`` avoids dividing by zero."""
    mean = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True)
    return (beats - mean) / (std + eps)


def maxabs(beats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale each beat so its largest absolute value is 1."""
    return beats / (np.abs(beats).max(axis=1, keepdims=True) + eps)


def clip(beats: np.ndarray, limit: float) -> np.ndarray:
    """Cut off values above ``limit`` and below ``-limit``, so rare spikes cannot dominate."""
    return np.clip(beats, -limit, limit)


def normalize_beats(beats: np.ndarray, cfg: dict) -> np.ndarray:
    """Run the method chosen in the ``normalize:`` block of configs/preprocess.yaml."""
    n = cfg["normalize"]
    method = n.get("method", "zscore")
    if method == "zscore":
        beats = zscore(beats)
    elif method == "maxabs":
        beats = maxabs(beats)
    elif method != "none":
        raise ValueError(f"Unknown normalization: {method}")
    if n.get("clip"):  # optional
        beats = clip(beats, float(n["clip"]))
    return beats.astype(np.float32)  # float32 is what the model and the .npz files use
