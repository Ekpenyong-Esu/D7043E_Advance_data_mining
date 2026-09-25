"""Re-export of the dataset code, kept because the assignment asks for a ``src/dataset.py`` file.

The real code is in ``ecg.datasets``. ``noqa: F401`` tells the linter the unused imports are on purpose.
"""
from ecg.datasets.beat_dataset import BeatDataset  # noqa: F401
from ecg.datasets.loaders import build_loaders  # noqa: F401
