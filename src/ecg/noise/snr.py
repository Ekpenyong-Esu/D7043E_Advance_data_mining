"""Add noise to a clean signal at a target SNR. Pure numpy."""
import numpy as np


def power(x: np.ndarray) -> float:
    return float(np.mean(x**2))


def add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Tile / crop ``noise`` to ``signal`` length (random offset) and scale so that
    10*log10(P_signal / P_noise) == snr_db."""
    raise NotImplementedError
