"""Compact 1D CNN using only MAX78002-supported ops: Conv1d, pooling, ReLU, Linear.

When ai8x-training is importable, the same architecture is built from ai8x
fused layers (see ``ai8x_cnn.py``) so there is a single definition to maintain.
"""
import torch
from torch import nn


class ECGCNN(nn.Module):
    def __init__(self, channels: list[int], kernels: list[int], pools: list[int], num_classes: int = 4):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, 1, L)
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
