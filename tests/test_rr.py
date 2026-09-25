"""Check the RR-interval features."""
import numpy as np

from ecg.data.rr import rr_features


def test_premature_beat_has_short_pre_and_long_post_rr():
    fs = 360
    regular = np.arange(0, 20) * fs                   # one beat per second
    peaks = regular.copy(); peaks[10] -= fs // 2      # beat 10 arrives 0.5 s early
    f = rr_features(peaks, fs)
    assert f.shape == (20, 3)
    assert np.isclose(f[10, 0], 0.5) and np.isclose(f[10, 1], 1.5)   # short gap before, long gap after
    assert f[10, 2] < 0.6                             # pre-RR well below the local mean
    assert np.isclose(f[5, 2], 1.0)                   # a regular beat has ratio 1


def test_long_gap_is_capped():
    fs = 360
    peaks = np.array([0, 1, 2, 102, 103]) * fs        # a 100 s gap with no annotations
    f = rr_features(peaks, fs)
    assert f[3, 0] == 3.0 and f[2, 1] == 3.0          # the gap is capped at 3 s on both sides
    assert f[:, 0].max() <= 3.0
