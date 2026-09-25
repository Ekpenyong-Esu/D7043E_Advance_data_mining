"""One epoch of training, and prediction over a whole loader. No file reading or writing. Not written yet."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, loss_fn, optimizer, device) -> float:
    """Run one pass over the training data. Return the mean loss."""
    raise NotImplementedError


@torch.no_grad()  # no gradients needed when only predicting
def predict(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(y_true, y_pred)`` for every beat in the loader."""
    raise NotImplementedError
