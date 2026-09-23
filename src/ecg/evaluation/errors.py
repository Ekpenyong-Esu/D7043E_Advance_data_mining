"""Pick and plot representative misclassified beats (report §5.5)."""
import numpy as np


def select_examples(y_true: np.ndarray, y_pred: np.ndarray, pairs: list[tuple[int, int]], k: int) -> dict:
    """For each (true, pred) pair return up to ``k`` beat indices."""
    raise NotImplementedError


def plot_examples(X: np.ndarray, indices: dict, out_path: str) -> None:
    raise NotImplementedError
