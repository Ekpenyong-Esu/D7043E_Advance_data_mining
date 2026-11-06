"""The same CNN as ``cnn.py``, built with the ai8x layer wrappers from ai8x-training.

The wrappers behave like plain PyTorch layers in FP32 mode. They add what the
MAX78002 toolchain needs: the shape checks of the accelerator, quantisation-aware
training (``ai8x.initiate_qat``) and INT8 simulation of the quantised checkpoint.
``ecg.deployment.toolchain.import_ai8x`` must be called first; it puts the repo on the
path and selects the device.
"""
from torch import nn

from ecg.models.cnn import block_lengths


class AI87ECGNet(nn.Module):
    """1D CNN for the MAX78002: fused (MaxPool, Conv1d, ReLU) blocks and one wide Linear layer."""

    def __init__(self, ai8x, channels: list[int], kernels: list[int], pools: list[int], paddings: list[int],
                 num_classes: int = 4, input_length: int = 256, batchnorm: bool = False, dropout: float = 0.0,
                 bias: bool = True, **kwargs):
        super().__init__()
        lengths = block_lengths(input_length, kernels, pools, paddings)
        bn = "Affine" if batchnorm else None
        in_ch = 1
        for i, (out_ch, k, pool, pad) in enumerate(zip(channels, kernels, pools, paddings), start=1):
            if pool > 1:
                layer = ai8x.FusedMaxPoolConv1dReLU(in_ch, out_ch, k, pool_size=pool, pool_stride=pool,
                                                     padding=pad, bias=bias, batchnorm=bn, **kwargs)
            else:
                layer = ai8x.FusedConv1dReLU(in_ch, out_ch, k, padding=pad, bias=bias, batchnorm=bn, **kwargs)
            setattr(self, f"conv{i}", layer)
            in_ch = out_ch
        self.n_blocks = len(channels)
        self.dropout = nn.Dropout(dropout)  # training only; not part of the synthesised network
        # ``wide=True``: 32-bit output without activation, read back by the CPU and passed to softmax.
        self.fc = ai8x.Linear(in_ch * lengths[-1], num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        for i in range(1, self.n_blocks + 1):
            x = getattr(self, f"conv{i}")(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


def build_ai8x_cnn(ai8x, model_cfg: dict, input_length: int = 256) -> nn.Module:
    """Build the network from the ``model:`` block of a config."""
    m = {k: v for k, v in model_cfg.items() if k != "type"}
    return AI87ECGNet(ai8x, input_length=input_length, **m)
