"""Fixtures shared by all tests."""
import numpy as np
import pytest


@pytest.fixture
def rng():
    """A seeded random generator, so tests give the same result every run."""
    return np.random.default_rng(0)
