"""Class-balancing samplers (oversampling of S and F)."""
import numpy as np
from torch.utils.data import WeightedRandomSampler


def minority_oversampler(y: np.ndarray, power: float = 0.5) -> WeightedRandomSampler:
    """Sample weight per beat = (1 / class frequency) ** power."""
    raise NotImplementedError
