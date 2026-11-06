"""Check the activation-threshold bookkeeping for a chain of layers, with stand-in modules."""
import torch
from torch import nn

from ecg.training.qat import apply_thresholds_chain


class Layer(nn.Module):
    """Only the attributes ``apply_thresholds_chain`` touches."""

    def __init__(self, threshold: float, bias: float, wide: bool = False):
        super().__init__()
        self.op = nn.Linear(1, 1); self.op.bias = nn.Parameter(torch.tensor([bias]))
        self.activation_threshold = nn.Parameter(torch.tensor(threshold), requires_grad=False)
        self.final_scale = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.wide = wide


def test_chain_adjusts_bias_threshold_and_final_scale():
    model = nn.Module()
    model.conv1, model.conv2, model.fc = Layer(2.0, 1.0), Layer(3.0, 8.0), Layer(1.0, 4.0, wide=True)
    apply_thresholds_chain(model, ["conv1", "conv2", "fc"])
    assert model.conv2.op.bias.item() == 8.0 / 4          # divided by 2 ** threshold of conv1
    assert model.conv2.activation_threshold.item() == 1.0  # 3 - 2
    assert model.fc.op.bias.item() == 4.0 / 8              # divided by 2 ** original threshold of conv2
    assert model.fc.activation_threshold.item() == -2.0    # 1 - 3
    assert model.fc.final_scale.item() == 3.0              # wide layer: original minus adjusted threshold
    assert model.conv1.final_scale.item() == 0.0 and model.conv2.final_scale.item() == 0.0
