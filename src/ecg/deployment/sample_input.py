"""Export one preprocessed beat as the ``.npy`` sample that ``ai8xize.py --sample-input`` needs. Not written yet."""
import numpy as np


def export_sample(beat: np.ndarray, out_path: str) -> None:
    """Scale a (1, L) float beat in [-1, 1] to the int8 range [-128, 127] and save it."""
    raise NotImplementedError
