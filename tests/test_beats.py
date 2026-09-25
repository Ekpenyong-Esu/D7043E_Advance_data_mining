"""Check beat extraction."""
import numpy as np

from ecg.data.beats import extract_beats


def test_window_length_and_boundary_drop():
    sig = np.arange(1000, dtype=np.float32)          # fake signal: value equals index
    peaks = np.array([5, 100, 500, 995])             # first and last are too close to the edges
    windows, kept = extract_beats(sig, peaks, before=90, after=166)
    assert windows.shape == (2, 256)                 # two beats survive, each 90 + 166 samples
    assert kept.tolist() == [1, 2]
    assert windows[0, 90] == 100                     # the R-peak sits at index ``before``
    assert windows[1, 0] == 500 - 90                 # the window starts ``before`` samples earlier
