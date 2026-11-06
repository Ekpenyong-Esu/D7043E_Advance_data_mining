"""Wrap the processed arrays as a PyTorch Dataset, and scale beats to the model's input range.

The stored beats are z-scored and clipped to [-8, 8] (report section 3.4). The MAX78002
accelerator takes inputs in [-1, 1), so the last step before the model divides by 8.
In INT8 simulation the same values are integers in [-128, 127], as on the chip.
"""
from collections.abc import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

Transform = Callable[[np.ndarray], np.ndarray]


def to_model_input(X: np.ndarray, input_scale: float, int8: bool = False) -> np.ndarray:
    """Map stored beats to the model input range.

    ``int8=False``: float values in [-1, 127/128], what the FP32 model and QAT see.
    ``int8=True``: the same values as integers in [-128, 127], what the quantised model and the chip see.
    """
    x = np.clip(np.asarray(X, dtype=np.float32) / input_scale, -1.0, 127.0 / 128.0)
    if int8:
        x = np.clip(np.round(x * 128.0), -128, 127)
    return x.astype(np.float32)


class BeatDataset(Dataset):
    """One item = (beat tensor of shape (1, L), class index)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, input_scale: float,
                 transform: Transform | None = None, int8: bool = False):
        self.X, self.y, self.input_scale, self.transform, self.int8 = X, y, input_scale, transform, int8

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = self.X[i]
        if self.transform is not None:   # training only
            x = self.transform(x)
        x = to_model_input(x, self.input_scale, self.int8)
        return torch.from_numpy(np.ascontiguousarray(x)), int(self.y[i])
