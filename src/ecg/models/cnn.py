"""The 1D CNN in plain PyTorch, and the layer table for report section 4.2.

Only layers the MAX78002 accelerator supports are used: MaxPool1d, Conv1d, ReLU and
one Linear output layer. Every pool is placed in front of a conv, because the chip fuses
"pool then conv then ReLU" into one layer. ``ai8x_cnn.py`` builds the same network with
the ai8x wrappers; the two must keep the same layer names and shapes.
"""
import pandas as pd
import torch
from torch import nn


def block_lengths(input_length: int, kernels: list[int], pools: list[int], paddings: list[int]) -> list[int]:
    """Output length of each conv block, for a stride-1 conv after an optional max-pool."""
    out, L = [], input_length
    for k, pool, pad in zip(kernels, pools, paddings):
        L = L // pool if pool > 1 else L
        L = L + 2 * pad - k + 1
        out.append(L)
    return out


class ECGCNN(nn.Module):
    def __init__(self, channels: list[int], kernels: list[int], pools: list[int], paddings: list[int],
                 num_classes: int = 4, input_length: int = 256, batchnorm: bool = False, dropout: float = 0.0, **_):
        super().__init__()
        self.lengths = block_lengths(input_length, kernels, pools, paddings)
        in_ch = 1
        for i, (out_ch, k, pool, pad) in enumerate(zip(channels, kernels, pools, paddings), start=1):
            steps: list[nn.Module] = [nn.MaxPool1d(pool)] if pool > 1 else []
            steps.append(nn.Conv1d(in_ch, out_ch, k, padding=pad))
            if batchnorm:
                steps.append(nn.BatchNorm1d(out_ch))
            steps.append(nn.ReLU())
            setattr(self, f"conv{i}", nn.Sequential(*steps))
            in_ch = out_ch
        self.n_blocks = len(channels)
        self.dropout = nn.Dropout(dropout)  # training only; the identity at inference, so nothing for the chip to run
        self.fc = nn.Linear(in_ch * self.lengths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` has shape (batch, 1, L). Returns class scores of shape (batch, num_classes)."""
        for i in range(1, self.n_blocks + 1):
            x = getattr(self, f"conv{i}")(x)
        return self.fc(self.dropout(x.flatten(1)))


def count_parameters(model: nn.Module) -> int:
    """Number of trainable weights and biases."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_table(m: dict, input_length: int = 256) -> pd.DataFrame:
    """One row per accelerator layer: input shape, operation, output shape, parameters.

    Computed from the ``model:`` block of the config, so it needs no trained weights.
    """
    lengths = block_lengths(input_length, m["kernels"], m["pools"], m["paddings"])
    rows, in_ch, L_in = [], 1, input_length
    for i, (out_ch, k, pool, pad, L_out) in enumerate(zip(m["channels"], m["kernels"], m["pools"], m["paddings"], lengths), 1):
        op = (f"MaxPool {pool}, " if pool > 1 else "") + f"Conv1d k={k} p={pad}, ReLU"
        rows.append({"Layer": f"conv{i}", "Input": f"{in_ch} x {L_in}", "Operation": op,
                     "Output": f"{out_ch} x {L_out}", "Parameters": out_ch * in_ch * k + out_ch})
        in_ch, L_in = out_ch, L_out
    flat = in_ch * L_in
    rows.append({"Layer": "fc", "Input": f"{flat} (flattened)", "Operation": "Linear",
                 "Output": str(m["num_classes"]), "Parameters": flat * m["num_classes"] + m["num_classes"]})
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Layer": "Total", "Input": "", "Operation": "", "Output": "", "Parameters": int(df["Parameters"].sum())}
    return df
