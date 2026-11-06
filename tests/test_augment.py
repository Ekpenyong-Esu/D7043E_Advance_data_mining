"""Check the training-only augmentations keep the beat shape and size."""
import numpy as np

from ecg.datasets.augment import (
    amplitude_scale,
    baseline_wander,
    build_transform,
    gaussian_noise,
    time_shift,
)


def test_time_shift_moves_the_peak(rng):
    x = np.zeros((1, 20), dtype=np.float32); x[0, 10] = 5.0
    y = time_shift(x, 3, rng)
    assert y.shape == x.shape and y.max() == 5.0            # the peak survives
    assert abs(int(np.argmax(y[0])) - 10) <= 3               # and moved by at most 3 samples


def test_scale_noise_wander_change_little(rng):
    x = np.ones((1, 256), dtype=np.float32)
    assert 0.8 <= amplitude_scale(x, 0.8, 1.2, rng).mean() <= 1.2
    assert abs(gaussian_noise(x, 0.02, rng).std() - 0.02) < 0.01
    assert np.abs(baseline_wander(x, 360, 0.3, rng) - x).max() <= 0.3


def test_build_transform_off_and_on():
    assert build_transform({"enabled": False}, 360, 0) is None
    t = build_transform({"amplitude_scale": [0.8, 1.2], "shift_samples": 5, "gaussian_noise_std": 0.02,
                         "baseline_wander_amplitude": 0.3}, 360, 0)
    x = np.random.default_rng(1).normal(size=(1, 256)).astype(np.float32)
    y = t(x)
    assert y.shape == x.shape and y.dtype == np.float32 and not np.array_equal(x, y)
