"""Random changes applied to training beats only, to make the model more robust.

Each function takes one beat of shape (1, L) and returns a changed copy.
Beats are never stretched in time, because timing carries class information.
Not written yet.
"""
import numpy as np


def amplitude_scale(x: np.ndarray, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    """Multiply the beat by a random factor between ``low`` and ``high``."""
    raise NotImplementedError


def time_shift(x: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    """Move the beat left or right by up to ``max_shift`` samples."""
    raise NotImplementedError


def gaussian_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add random noise with standard deviation ``std``."""
    raise NotImplementedError


def baseline_wander(x: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    """Add a slow sine wave, like the drift caused by breathing."""
    raise NotImplementedError


def build_transform(cfg: dict, seed: int):
    """Read ``cfg['augmentation']`` and return one function that applies the enabled augmentations."""
    raise NotImplementedError
