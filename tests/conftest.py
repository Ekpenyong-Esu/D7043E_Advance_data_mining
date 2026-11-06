"""Fixtures shared by all tests."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """A seeded random generator, so tests give the same result every run."""
    return np.random.default_rng(0)


@pytest.fixture
def model_cfg():
    """The ``model:`` block of configs/model.yaml, written out so the tests do not depend on the file."""
    return {"type": "cnn1d", "channels": [16, 32, 64, 64], "kernels": [5, 5, 3, 3], "paddings": [2, 2, 1, 1],
            "pools": [1, 2, 2, 4], "batchnorm": False, "dropout": 0.0, "num_classes": 4}
