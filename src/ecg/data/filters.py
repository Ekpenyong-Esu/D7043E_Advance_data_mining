"""Filters applied to the whole record, before it is cut into beats.

The main job is to remove baseline wander: the slow up-and-down drift caused
by breathing and electrode movement. An optional low-pass filter can also
remove fast noise such as muscle activity.
"""
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, filtfilt


def _odd(n: int) -> int:
    """Median filters need an odd window length. Round up if ``n`` is even."""
    return n if n % 2 == 1 else n + 1


def remove_baseline_median(x: np.ndarray, fs: int, kernels_ms: tuple[int, int]) -> np.ndarray:
    """Estimate the baseline with two median filters in a row, then subtract it.

    The first filter (about 200 ms) removes the QRS spikes. The second
    (about 600 ms) removes the P and T waves. What is left is the baseline.
    """
    baseline = x
    for ms in kernels_ms:
        baseline = median_filter(baseline, size=_odd(round(ms * fs / 1000)), mode="nearest")
    return x - baseline


def highpass(x: np.ndarray, fs: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    """Butterworth high-pass filter. Removes slow drift below ``cutoff_hz``.

    ``filtfilt`` runs the filter forwards and backwards, so the peaks do not shift in time.
    """
    b, a = butter(order, cutoff_hz / (fs / 2), btype="high")  # cutoff is given as a fraction of Nyquist
    return filtfilt(b, a, x)


def lowpass(x: np.ndarray, fs: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """Butterworth low-pass filter. Removes fast noise above ``cutoff_hz`` (muscle, mains hum)."""
    b, a = butter(order, cutoff_hz / (fs / 2), btype="low")
    return filtfilt(b, a, x)


def apply_filters(x: np.ndarray, fs: int, cfg: dict) -> np.ndarray:
    """Run the filters chosen in the ``filter:`` block of configs/preprocess.yaml."""
    f = cfg["filter"]
    if not f.get("enabled", True):
        return x
    method = f.get("method", "median")
    if method == "median":
        x = remove_baseline_median(x, fs, tuple(f["median_kernels_ms"]))
    elif method == "highpass":
        x = highpass(x, fs, f["highpass_hz"])
    elif method != "none":
        raise ValueError(f"Unknown filter method: {method}")
    if f.get("lowpass_hz"):  # optional extra step
        x = lowpass(x, fs, f["lowpass_hz"])
    return x
