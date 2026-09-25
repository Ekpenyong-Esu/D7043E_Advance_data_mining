"""Samplers that draw the rare classes (S and F) more often during training. Not written yet."""
import numpy as np
from torch.utils.data import WeightedRandomSampler


def minority_oversampler(y: np.ndarray, power: float = 0.5) -> WeightedRandomSampler:
    """Give each beat the weight (1 / frequency of its class) ** power, so rare classes are drawn more."""
    raise NotImplementedError
