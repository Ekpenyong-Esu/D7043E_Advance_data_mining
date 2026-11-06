"""Add noise to a clean signal at a chosen signal-to-noise ratio (SNR).

SNR in dB = 10 * log10(power of signal / power of noise).
0 dB means the noise is as strong as the signal. 12 dB means the signal is 16 times stronger.
Power is measured after removing the mean, so a constant offset does not count as signal.
"""
import numpy as np


def power(x: np.ndarray) -> float:
    """Mean squared value around the mean (the variance)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean((x - x.mean()) ** 2))


def tile_noise(noise: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Cut ``length`` samples of noise, starting at a random offset and wrapping around if needed."""
    start = int(rng.integers(0, len(noise)))
    idx = (start + np.arange(length)) % len(noise)
    return noise[idx]


def add_noise(signal: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Return ``signal`` plus noise scaled so that the SNR equals ``snr_db``."""
    n = tile_noise(noise, len(signal), rng)
    n = n - n.mean()
    scale = np.sqrt(power(signal) / (power(n) * 10 ** (snr_db / 10)))
    return signal + scale * n
