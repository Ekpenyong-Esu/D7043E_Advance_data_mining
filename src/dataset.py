"""Entry point / re-export: PyTorch datasets and loaders.

Kept for the required project layout. Implementation lives in ``ecg.datasets``.
"""
from ecg.datasets.beat_dataset import BeatDataset  # noqa: F401
from ecg.datasets.loaders import build_loaders  # noqa: F401
