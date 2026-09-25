"""Add noise to a clean signal at a chosen signal-to-noise ratio (SNR). Not written yet.

SNR in dB = 10 * log10(power of signal / power of noise).
0 dB means the noise is as strong as the signal. 12 dB means the signal is much stronger.
"""
import numpy as np


def power(x: np.ndarray) -> float:
    """Mean of the squared values."""
    return float(np.mean(x**2))


def add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``signal`` plus scaled noise.

    The noise is repeated or cut to the signal length, starting at a random offset,
    then scaled so that the SNR equals ``snr_db``.
    """
    raise NotImplementedError
