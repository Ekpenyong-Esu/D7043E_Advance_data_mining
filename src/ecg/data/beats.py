"""Cut a fixed-length window around each R-peak.

The model needs every input to have the same length. So for each annotated
beat we take a fixed number of samples before the R-peak and after it.
"""
import numpy as np


def extract_beats(
    signal: np.ndarray, r_peaks: np.ndarray, before: int, after: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(windows, kept_indices)``.

    ``windows`` has shape (n_kept, before + after). Row i is the beat around
    ``r_peaks[kept_indices[i]]``, and the R-peak sits at column ``before``.

    A beat whose window would run past the start or end of the signal is dropped.
    ``kept_indices`` says which peaks survived, so labels can be lined up with the windows.
    """
    r_peaks = np.asarray(r_peaks, dtype=np.int64)
    # A peak is kept only if the whole window fits inside the signal.
    keep = (r_peaks - before >= 0) & (r_peaks + after <= len(signal))
    kept_idx = np.flatnonzero(keep)
    # Build all windows at once: each row is peak + [-before, ..., after - 1].
    offsets = np.arange(-before, after)
    windows = signal[r_peaks[kept_idx][:, None] + offsets[None, :]]
    return windows.astype(np.float32), kept_idx
