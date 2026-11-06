"""Export one preprocessed beat as the ``.npy`` sample that ``ai8xize.py --sample-input`` needs.

The synthesiser runs this beat through its own model of the accelerator, writes the input
into ``sampledata.h`` and the expected output into the generated C code as a known-answer
test. The values must be the integers the chip sees: the stored beat divided by the clip
limit, scaled to [-128, 127].
"""
from pathlib import Path

import numpy as np

from ecg.datasets.beat_dataset import to_model_input


def export_sample(beat: np.ndarray, input_scale: float, path: str | Path) -> np.ndarray:
    """Save a stored beat of shape (1, L) as int64 values in [-128, 127]. Returns the saved array."""
    x = to_model_input(beat[None], input_scale, int8=True)[0].astype(np.int64)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, x)
    return x
