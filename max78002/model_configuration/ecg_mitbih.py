###################################################################################################
# ECG beat dataset for ai8x-training, D7043E Group 16.
#
# Copy this file into ai8x-training/datasets/ and point it at the project's data/processed_w512/ (the 512-sample window of the final model).
# It serves the beats written by notebooks/EDA.ipynb, scaled to the accelerator input range
# exactly as src/ecg/datasets/beat_dataset.py does: stored value / 8, clipped to [-1, 127/128]
# (or the same as integers in [-128, 127] when train.py runs with -8 / --act-mode-8bit).
###################################################################################################
"""MIT-BIH heartbeat dataset (N, S, V, F) for the ai8x training framework."""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

import ai8x

INPUT_SCALE = 8.0   # the clip limit of configs/preprocess.yaml


class ECGBeats(Dataset):
    """One item = (beat of shape (1, 512), class index)."""

    def __init__(self, root, split, transform=None):
        d = np.load(os.path.join(root, f"{split}.npz"))
        self.X, self.y, self.transform = d["X"], d["y"], transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        # ai8x.normalize expects values in [0, 1]; map [-1, 1) there first.
        x = np.clip(self.X[i] / INPUT_SCALE, -1.0, 127.0 / 128.0).astype(np.float32)
        x = torch.from_numpy((x + 1.0) / 2.0)
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.y[i])


def ecg_mitbih_get_datasets(data, load_train=True, load_test=True):
    """Return (train, validation) datasets. The validation split stands in for ai8x's test split;
    DS2 (test.npz) must stay untouched until the final evaluation."""
    (data_dir, args) = data
    transform = ai8x.normalize(args=args)
    root = os.path.join(data_dir, "ECG_MITBIH")
    train = ECGBeats(root, "train", transform) if load_train else None
    val = ECGBeats(root, "val", transform) if load_test else None
    return train, val


datasets = [
    {
        'name': 'ecg_mitbih',
        'input': (1, 512),
        'output': (0, 1, 2, 3),
        'loader': ecg_mitbih_get_datasets,
    },
]
