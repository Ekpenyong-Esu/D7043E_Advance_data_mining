"""Loss functions and class weights for the imbalanced data. Report section 3.6."""
import numpy as np
import torch
from torch import nn


def class_weights(y: np.ndarray, num_classes: int, power: float) -> torch.Tensor:
    """Weight for class c = (N / count_c) ** power, scaled so the mean weight is 1.

    ``power = 1`` is full inverse frequency, ``power = 0.5`` the inverse square root,
    which emphasises rare classes without making their gradients unstable.
    """
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    w = (counts.sum() / np.maximum(counts, 1)) ** power
    return torch.as_tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Cross-entropy that pays less attention to easy, confidently classified beats.

    loss = -(1 - p_true) ** gamma * log(p_true), optionally class-weighted.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.ones(0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        focal = (1.0 - log_p.exp()) ** self.gamma * (-log_p)
        if self.weight.numel():
            focal = focal * self.weight[target]
        return focal.mean()


def build_loss(tr_cfg: dict, y_train: np.ndarray, num_classes: int) -> nn.Module:
    """Return the loss named in ``tr_cfg['loss']``: ``ce``, ``weighted_ce`` or ``focal``."""
    kind = tr_cfg.get("loss", "weighted_ce")
    if kind == "ce":
        return nn.CrossEntropyLoss()
    w = class_weights(y_train, num_classes, float(tr_cfg.get("class_weight_power", 0.5)))
    if kind == "weighted_ce":
        return nn.CrossEntropyLoss(weight=w)
    if kind == "focal":
        return FocalLoss(float(tr_cfg.get("focal_gamma", 2.0)), w)
    raise ValueError(f"Unknown loss: {kind}")
