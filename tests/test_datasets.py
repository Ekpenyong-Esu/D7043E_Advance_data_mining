"""Check the input scaling, the dataset and the minority sampler."""
import numpy as np
import pytest

from ecg.datasets.beat_dataset import BeatDataset, to_model_input
from ecg.datasets.loaders import DS2Locked, load_split
from ecg.datasets.samplers import class_sample_weights


def test_model_input_range():
    x = np.array([[[-8.0, 0.0, 8.0, 100.0]]], dtype=np.float32)
    f = to_model_input(x, 8.0)
    assert f.min() == -1.0 and f.max() <= 127 / 128           # chip range, the top clipped just below 1
    i = to_model_input(x, 8.0, int8=True)
    assert i.tolist() == [[[-128.0, 0.0, 127.0, 127.0]]]        # the same as integers


def test_dataset_item_shape():
    X = np.zeros((3, 1, 256), dtype=np.float32); y = np.array([0, 1, 2])
    x, label = BeatDataset(X, y, 8.0)[1]
    assert tuple(x.shape) == (1, 256) and label == 1


def test_sampler_weights_favour_rare_classes():
    y = np.array([0] * 90 + [1] * 10)
    w = class_sample_weights(y, 0.5)
    assert w[-1] > w[0] and np.isclose(w.mean(), 1.0)
    assert np.allclose(class_sample_weights(y, 0.0), 1.0)      # power 0 = no oversampling


def test_test_split_is_locked(tmp_path):
    with pytest.raises(DS2Locked):
        load_split(tmp_path, "test")
