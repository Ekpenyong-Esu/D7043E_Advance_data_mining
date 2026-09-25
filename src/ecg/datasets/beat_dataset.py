"""Wrap the processed arrays as a PyTorch Dataset.

Augmentation is passed in as a function (``transform``). The dataset does not
know which augmentations exist, so the same class serves train, val and test.
"""
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

Transform = Callable[[np.ndarray], np.ndarray]  # a function that changes one beat and returns it


class BeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Transform | None = None):
        """``X`` has shape (n, 1, L), ``y`` has shape (n,). ``transform`` is optional."""
        self.X, self.y, self.transform = X, y, transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        """Return beat ``i`` as a float32 tensor and its label as an int."""
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.y[i])
