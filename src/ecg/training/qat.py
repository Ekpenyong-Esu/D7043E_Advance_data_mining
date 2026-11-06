"""Prepare the ai8x network for quantisation-aware training (QAT).

Post-training quantisation clamps every activation to [-1, 1) with 8-bit steps, which
loses most of the signal when the FP32 activations are much larger. ai8x's answer is a
power-of-two scale per layer, found from activation histograms on training data
(``pre_qat`` in ai8x-training). Its graph-tracing step does not run on the PyTorch version
used here, so the scale bookkeeping for a plain chain of layers is done in this module
with the same formulas: each layer's bias and threshold are adjusted by the threshold of
the layer before it, and only the output layer keeps a final scale.
"""
from types import SimpleNamespace

import torch
from torch import nn


def apply_thresholds_chain(model: nn.Module, names: list[str]) -> None:
    """ai8x ``apply_scales`` for a network whose layers ``names`` feed each other in order."""
    mods = [getattr(model, n) for n in names]
    original = [m.activation_threshold.detach().clone() for m in mods]
    for i in range(1, len(mods)):
        prev, cur, t_prev = mods[i - 1], mods[i], original[i - 1]
        if cur.op is not None and cur.op.bias is not None:
            cur.op.bias = nn.Parameter(cur.op.bias / torch.exp2(t_prev))
        cur.activation_threshold = nn.Parameter(cur.activation_threshold - t_prev, requires_grad=False)
        scale = original[i] - cur.activation_threshold if cur.wide else original[i]
        cur.final_scale = nn.Parameter(scale.detach().clone(), requires_grad=False)
        prev.final_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)


def prepare_qat(ai8x, model: nn.Module, loader, device, policy: dict, layer_names: list[str]) -> None:
    """Measure activation ranges, set the per-layer scales, then switch the layers to QAT mode."""
    if policy.get("activation_thresholds", True):
        ai8x.init_hist(model)
        ai8x.stat_collect(loader, model, SimpleNamespace(device=device))
        ai8x.init_threshold(model, policy.get("outlier_removal_z_score", 8.0))
        ai8x.release_hist(model)
        apply_thresholds_chain(model, layer_names)
    ai8x.initiate_qat(model, policy)
