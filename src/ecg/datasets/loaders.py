"""Build train / val / test DataLoaders from a config."""
from torch.utils.data import DataLoader


def build_loaders(cfg: dict) -> dict[str, DataLoader]:
    """Train loader gets augmentation + sampler; val/test get neither."""
    raise NotImplementedError
