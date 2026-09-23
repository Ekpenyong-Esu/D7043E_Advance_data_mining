"""One epoch of training / evaluation. No file IO, no config parsing."""
import numpy as np
import torch
from torch.utils.data import DataLoader


def train_one_epoch(model, loader: DataLoader, loss_fn, optimizer, device) -> float:
    """Return mean training loss."""
    raise NotImplementedError


@torch.no_grad()
def predict(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_pred) for a whole loader."""
    raise NotImplementedError
