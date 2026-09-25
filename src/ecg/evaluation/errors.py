"""Pick and draw beats the model got wrong, for report section 5.5. Not written yet."""
import numpy as np


def select_examples(y_true: np.ndarray, y_pred: np.ndarray, pairs: list[tuple[int, int]], k: int) -> dict:
    """For each (true, predicted) pair, for example (N, S), return up to ``k`` beat indices."""
    raise NotImplementedError


def plot_examples(X: np.ndarray, indices: dict, out_path: str) -> None:
    """Draw the chosen beats, one panel per pair, and save the figure."""
    raise NotImplementedError
