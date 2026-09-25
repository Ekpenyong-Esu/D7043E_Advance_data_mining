"""Loss functions and class weights for the imbalanced data. Not written yet."""
import numpy as np
import torch
from torch import nn


def class_weights(y: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    """Weight for class c = (N / count_c) ** power, then scaled so the mean weight is 1.

    Rare classes get a larger weight, so mistakes on them cost more.
    """
    raise NotImplementedError


class FocalLoss(nn.Module):
    """Cross-entropy that pays less attention to easy, confident examples."""

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        raise NotImplementedError


def build_loss(cfg: dict, y_train: np.ndarray) -> nn.Module:
    """Return the loss chosen in ``cfg['training']``, with class weights computed from ``y_train``."""
    raise NotImplementedError
