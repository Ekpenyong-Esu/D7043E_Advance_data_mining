"""Build the train, validation and test DataLoaders from a config. Not written yet."""
from torch.utils.data import DataLoader


def build_loaders(cfg: dict) -> dict[str, DataLoader]:
    """Return ``{"train": ..., "val": ..., "test": ...}``.

    Only the train loader gets augmentation and the oversampler. Val and test are left as they are.
    """
    raise NotImplementedError
