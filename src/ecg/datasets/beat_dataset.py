"""Wrap processed arrays as a torch Dataset. Augmentation is injected, not hard-coded."""
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

Transform = Callable[[np.ndarray], np.ndarray]


class BeatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, transform: Transform | None = None):
        self.X, self.y, self.transform = X, y, transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = self.X[i]
        if self.transform is not None:
            x = self.transform(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.y[i])
