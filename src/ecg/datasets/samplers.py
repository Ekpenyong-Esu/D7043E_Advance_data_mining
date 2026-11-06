"""Draw the rare classes (S and F) more often during training. Report section 3.6."""
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def class_sample_weights(y: np.ndarray, power: float) -> np.ndarray:
    """Weight of each beat = (1 / frequency of its class) ** power.

    ``power = 0`` gives every beat the same weight (no oversampling).
    ``power = 1`` makes every class equally likely in a batch.
    ``power = 0.5`` is in between: rare classes are drawn more often, but N still dominates.
    """
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64)
    freq = counts / counts.sum()
    w = (1.0 / freq[y]) ** power
    return w / w.mean()


def minority_oversampler(y: np.ndarray, power: float, seed: int) -> WeightedRandomSampler:
    """A sampler that draws ``len(y)`` beats per epoch, with replacement, using the weights above."""
    g = torch.Generator(); g.manual_seed(seed)
    w = torch.as_tensor(class_sample_weights(y, power), dtype=torch.double)
    return WeightedRandomSampler(w, num_samples=len(y), replacement=True, generator=g)
