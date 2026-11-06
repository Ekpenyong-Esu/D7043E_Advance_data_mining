"""Check that noise is added at the requested SNR."""
import numpy as np

from ecg.noise.snr import add_noise, power


def test_target_snr_is_reached(rng):
    t = np.arange(36000) / 360
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.5          # a clean wave with an offset
    noise = rng.normal(size=100000)
    for snr in (12, 6, 0):
        noisy = add_noise(signal, noise, snr, rng)
        measured = 10 * np.log10(power(signal) / power(noisy - signal))
        assert abs(measured - snr) < 0.1
