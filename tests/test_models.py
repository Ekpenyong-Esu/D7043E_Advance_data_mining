"""Check the CNN shapes, the parameter count and the layer table."""
import torch

from ecg.models.cnn import ECGCNN, block_lengths, count_parameters, layer_table


def test_block_lengths():
    assert block_lengths(256, [5, 5, 3, 3], [1, 2, 2, 4], [2, 2, 1, 1]) == [256, 128, 64, 16]


def test_cnn_output_and_parameter_count(model_cfg):
    m = ECGCNN(**{k: v for k, v in model_cfg.items() if k != "type"})
    assert m(torch.zeros(2, 1, 256)).shape == (2, 4)
    t = layer_table(model_cfg)
    assert count_parameters(m) == int(t.loc[t["Layer"] == "Total", "Parameters"].iloc[0]) == 25348
    assert count_parameters(m) < 100_000                       # success criterion of section 1.4


def test_cnn_uses_only_chip_operations(model_cfg):
    m = ECGCNN(**{k: v for k, v in model_cfg.items() if k != "type"})
    allowed = (torch.nn.Conv1d, torch.nn.MaxPool1d, torch.nn.ReLU, torch.nn.Linear, torch.nn.Sequential,
               torch.nn.Dropout, torch.nn.BatchNorm1d, ECGCNN)
    assert all(isinstance(mod, allowed) for mod in m.modules())
