"""Loss functions and class-weight computation."""
import numpy as np
import torch
from torch import nn


def class_weights(y: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    """weight_c = (N / count_c) ** power, normalized to mean 1."""
    raise NotImplementedError


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        raise NotImplementedError


def build_loss(cfg: dict, y_train: np.ndarray) -> nn.Module:
    raise NotImplementedError
