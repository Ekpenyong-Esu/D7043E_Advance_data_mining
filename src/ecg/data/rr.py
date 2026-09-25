"""Timing features from the R-peak positions.

An S beat looks almost like an N beat. The difference is timing: an S beat
comes early, so the gap before it is short and the gap after it is long.
These gaps are called RR intervals. This module computes them so a model
that sees only one beat still knows about its timing.
"""
import numpy as np

# Gaps longer than this are not real RR intervals. They are stretches with no
# annotations (for example flutter in record 207). We cap them to keep the values sane.
MAX_RR_S = 3.0


def rr_features(r_peaks: np.ndarray, fs: int, local_window: int = 10, max_rr_s: float = MAX_RR_S) -> np.ndarray:
    """Return an array of shape (n_beats, 3) with, for each beat:

      column 0  pre_rr   seconds since the previous beat
      column 1  post_rr  seconds until the next beat
      column 2  ratio    pre_rr divided by the mean pre_rr of the last ``local_window`` beats

    The first beat has no previous beat and the last has no next beat.
    They copy the nearest valid value.
    """
    r = np.asarray(r_peaks, dtype=np.float64) / fs  # peak positions in seconds
    n = len(r)
    if n < 2:
        return np.zeros((n, 3), dtype=np.float32)

    # Gap to the previous beat. The first beat copies the second one's value.
    pre = np.empty(n); pre[1:] = np.diff(r); pre[0] = pre[1]
    # Gap to the next beat. The last beat copies the one before it.
    post = np.empty(n); post[:-1] = np.diff(r); post[-1] = post[-2]
    # Cap long gaps.
    np.minimum(pre, max_rr_s, out=pre); np.minimum(post, max_rr_s, out=post)

    # Running mean of the previous ``local_window`` pre-RR values.
    # We pad the front with the first value so the early beats also get a mean.
    kernel = np.ones(local_window) / local_window
    padded = np.concatenate([np.full(local_window, pre[0]), pre[:-1]])
    local_mean = np.convolve(padded, kernel, mode="valid")[:n]

    ratio = pre / np.maximum(local_mean, 1e-6)  # below 1 means the beat came early
    return np.stack([pre, post, ratio], axis=1).astype(np.float32)
