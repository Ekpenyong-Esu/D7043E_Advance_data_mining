"""Check normalisation and clipping."""
import numpy as np

from ecg.data.normalize import normalize_beats


def test_zscore_and_clip():
    beats = np.array([[0, 1, 2, 3, 100.0]])          # one beat with a big outlier
    cfg = {"normalize": {"method": "zscore", "clip": 1.5}}
    out = normalize_beats(beats, cfg)
    assert out.dtype == np.float32
    assert out.max() <= 1.5 and out.min() >= -1.5   # the outlier was clipped
