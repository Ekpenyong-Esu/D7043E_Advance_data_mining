###################################################################################################
# ECG heartbeat classifier for the MAX78002, D7043E Group 16.
#
# Copy this file into ai8x-training/models/ to train the same network with the official
# train.py (python train.py --model ai87ecgnet --dataset ecg_mitbih --device MAX78002 ...).
# The project itself trains the network from src/ecg/models/ai8x_cnn.py with the same ai8x
# layers, so the checkpoints are interchangeable. Keep the layer sizes in sync with
# configs/model.yaml (see results/model_layers.csv for the final values).
###################################################################################################
"""ECG beat classifier: fused (MaxPool1d, Conv1d, ReLU) blocks and one wide Linear layer."""
from torch import nn

import ai8x

CHANNELS = [16, 32, 64, 64]
KERNELS = [5, 5, 3, 3]
PADDINGS = [2, 2, 1, 1]
POOLS = [1, 2, 4, 4]        # final model: 512-sample window, see configs/preprocess_w512.yaml


class AI87ECGNet(nn.Module):
    """1D CNN for single ECG beats (1 x 512 input, 4 classes), BatchNorm folded before quantisation."""

    def __init__(self, num_classes=4, num_channels=1, dimensions=(512, 1), bias=True, batchnorm="Affine", **kwargs):
        super().__init__()
        length, in_ch = dimensions[0], num_channels
        for i, (out_ch, k, pad, pool) in enumerate(zip(CHANNELS, KERNELS, PADDINGS, POOLS), start=1):
            if pool > 1:
                layer = ai8x.FusedMaxPoolConv1dReLU(in_ch, out_ch, k, pool_size=pool, pool_stride=pool,
                                                     padding=pad, bias=bias, batchnorm=batchnorm, **kwargs)
                length //= pool
            else:
                layer = ai8x.FusedConv1dReLU(in_ch, out_ch, k, padding=pad, bias=bias, batchnorm=batchnorm, **kwargs)
            length = length + 2 * pad - k + 1
            setattr(self, f"conv{i}", layer)
            in_ch = out_ch
        self.dropout = nn.Dropout(0.0)
        self.fc = ai8x.Linear(in_ch * length, num_classes, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        for i in range(1, len(CHANNELS) + 1):
            x = getattr(self, f"conv{i}")(x)
        x = x.view(x.size(0), -1)
        return self.fc(self.dropout(x))


def ai87ecgnet(pretrained=False, **kwargs):
    """Constructs an AI87ECGNet model."""
    assert not pretrained
    return AI87ECGNet(**kwargs)


models = [
    {
        'name': 'ai87ecgnet',
        'min_input': 1,
        'dim': 1,
    },
]
