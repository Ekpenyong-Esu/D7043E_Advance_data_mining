"""Export one preprocessed beat as the .npy sample ai8xize.py needs (int8-scaled)."""
import numpy as np


def export_sample(beat: np.ndarray, out_path: str) -> None:
    """Scale a (1, L) float beat in [-1, 1] to int8 range and save."""
    raise NotImplementedError
