"""The 1D CNN, in plain PyTorch. Not written yet.

It uses only layers the MAX78002 accelerator supports: Conv1d, pooling, ReLU and Linear.
``ai8x_cnn.py`` builds the same network with ai8x layers. The two must stay identical.
"""
import torch
from torch import nn


class ECGCNN(nn.Module):
    def __init__(self, channels: list[int], kernels: list[int], pools: list[int], num_classes: int = 4):
        """``channels``, ``kernels`` and ``pools`` have one entry per conv block."""
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape (batch, 1, L). Returns class scores of shape (batch, num_classes)."""
        raise NotImplementedError


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights. Reported in the model table of section 4.2."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
