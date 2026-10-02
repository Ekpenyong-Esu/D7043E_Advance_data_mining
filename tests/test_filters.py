"""Check the filters on made-up signals."""
import numpy as np

from ecg.data.filters import remove_baseline_median, lowpass


def test_median_baseline_removes_slow_drift():
    fs = 360; t = np.arange(fs * 10) / fs
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * t)         # slow wave, like baseline wander
    spikes = np.zeros_like(t); spikes[::fs] = 1.0      # one sharp "QRS" spike per second
    x = drift + spikes
    y = remove_baseline_median(x, fs, (200, 600))
    assert np.abs(y[fs // 2 :: fs]).max() < 0.05      # between spikes the drift is gone
    assert y[fs] > 0.9                                # the spike is still there


def test_lowpass_attenuates_high_frequency():
    fs = 360; t = np.arange(fs * 2) / fs
    x = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t)   # a 5 Hz wave plus a 100 Hz wave
    y = lowpass(x, fs, 40)
    hf = np.abs(np.fft.rfft(y))[int(100 * 2)]         # strength left at 100 Hz
    lf = np.abs(np.fft.rfft(y))[int(5 * 2)]           # strength left at 5 Hz
    assert hf < 0.05 * lf                             # the 100 Hz wave is almost gone
