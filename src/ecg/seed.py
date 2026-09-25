"""Set the random seed everywhere, so a run gives the same result each time."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and, if installed, PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)              # CPU
        torch.cuda.manual_seed_all(seed)     # every GPU
        # Make the GPU pick the same algorithm every run, even if it is not the fastest one.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # torch is not needed for the data pipeline
