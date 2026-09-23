"""Training-only augmentations. Each is a pure function on one beat (1, L)."""
import numpy as np


def amplitude_scale(x: np.ndarray, low: float, high: float, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError


def time_shift(x: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError


def gaussian_noise(x: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError


def baseline_wander(x: np.ndarray, fs: int, rng: np.random.Generator) -> np.ndarray:
    raise NotImplementedError


def build_transform(cfg: dict, seed: int):
    """Compose the augmentations enabled in ``cfg['augmentation']`` into one callable."""
    raise NotImplementedError
