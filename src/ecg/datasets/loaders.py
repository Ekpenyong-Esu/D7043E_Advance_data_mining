"""Read the processed splits and build the PyTorch loaders.

Only the training loader gets augmentation and the oversampler. Validation and test
beats are used exactly as stored. The test split is DS2: ``load_split`` refuses to read
it unless ``final=True`` is passed, so it cannot be used by accident during model selection.
"""
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from ecg.datasets.augment import build_transform
from ecg.datasets.beat_dataset import BeatDataset
from ecg.datasets.samplers import minority_oversampler


class DS2Locked(RuntimeError):
    """Raised when DS2 is requested without ``final=True``."""


def load_split(processed_dir: str | Path, split: str, final: bool = False) -> dict[str, np.ndarray]:
    """Return the arrays of one split: X (n, 1, L), y (n,), record_ids (n,), rr (n, 3)."""
    if split == "test" and not final:
        raise DS2Locked("DS2 is locked. Pass final=True only for the final evaluation.")
    with np.load(Path(processed_dir) / f"{split}.npz") as d:
        return {k: d[k] for k in d.files}


def build_loaders(cfg: dict, processed_dir: str | Path, splits=("train", "val")) -> dict[str, DataLoader]:
    """Build one DataLoader per split from a model config (``configs/model.yaml``)."""
    tr = cfg["training"]
    scale = float(cfg["input_scale"])
    loaders = {}
    for split in splits:
        d = load_split(processed_dir, split, final=(split == "test"))
        if split == "train":
            transform = build_transform(cfg.get("augmentation"), cfg["fs"], cfg["seed"])
            ds = BeatDataset(d["X"], d["y"], scale, transform)
            power = float(tr.get("oversample_power", 0) or 0)
            sampler = minority_oversampler(d["y"], power, cfg["seed"]) if power > 0 else None
            loaders[split] = DataLoader(ds, batch_size=tr["batch_size"], sampler=sampler,
                                        shuffle=sampler is None, drop_last=True)
        else:
            loaders[split] = DataLoader(BeatDataset(d["X"], d["y"], scale), batch_size=512, shuffle=False)
    return loaders
