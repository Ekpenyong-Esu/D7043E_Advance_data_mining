"""Random changes applied to training beats only, so the model sees more variety.

Each function takes one beat of shape (1, L) in normalised units (z-score, R-peak
around 5 to 7) and returns a changed copy. Beats are never stretched in time,
because the QRS width separates V beats from N beats. Report section 3.7.
"""
from collections.abc import Callable

import numpy as np

Transform = Callable[[np.ndarray], np.ndarray]


def amplitude_scale(x: np.ndarray, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply the beat by one random factor between ``low`` and ``high``."""
    return x * rng.uniform(low, high)


def time_shift(x: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Move the beat left or right by up to ``max_shift`` samples. The gap is filled with the edge value."""
    s = int(rng.integers(-max_shift, max_shift + 1))
    if s == 0:
        return x
    out = np.empty_like(x)
    if s > 0:   # shift right: the first s samples repeat the first value
        out[:, s:] = x[:, :-s]; out[:, :s] = x[:, :1]
    else:       # shift left: the last |s| samples repeat the last value
        out[:, :s] = x[:, -s:]; out[:, s:] = x[:, -1:]
    return out


def gaussian_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise with standard deviation ``std``."""
    return x + rng.normal(0.0, std, size=x.shape).astype(x.dtype)


def baseline_wander(x: np.ndarray, fs: int, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Add a slow sine wave (0.1 to 0.5 Hz), like the drift that breathing causes."""
    t = np.arange(x.shape[-1]) / fs
    f = rng.uniform(0.1, 0.5); a = rng.uniform(0.0, amplitude); phase = rng.uniform(0, 2 * np.pi)
    return x + (a * np.sin(2 * np.pi * f * t + phase)).astype(x.dtype)


def build_transform(aug_cfg: dict, fs: int, seed: int) -> Transform | None:
    """Read the ``augmentation:`` block of a model config and return one function that applies
    every enabled step in order, or None when augmentation is off."""
    if not aug_cfg or not aug_cfg.get("enabled", True):
        return None
    rng = np.random.default_rng(seed)
    scale = aug_cfg.get("amplitude_scale")
    shift = int(aug_cfg.get("shift_samples", 0) or 0)
    noise = float(aug_cfg.get("gaussian_noise_std", 0) or 0)
    wander = float(aug_cfg.get("baseline_wander_amplitude", 0) or 0)

    def transform(x: np.ndarray) -> np.ndarray:
        if scale:
            x = amplitude_scale(x, float(scale[0]), float(scale[1]), rng)
        if shift:
            x = time_shift(x, shift, rng)
        if wander:
            x = baseline_wander(x, fs, wander, rng)
        if noise:
            x = gaussian_noise(x, noise, rng)
        return x.astype(np.float32)

    return transform
