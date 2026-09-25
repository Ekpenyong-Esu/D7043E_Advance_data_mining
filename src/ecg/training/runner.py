"""Run one training job from start to end. Not written yet.

Builds the loaders, model and loss, trains for several epochs, checks the
validation score after each one, saves the best checkpoint and logs the run.
"""
from ecg.logging_utils import get_logger

log = get_logger(__name__)


def run_training(cfg: dict) -> None:
    """Train an sklearn baseline or a torch CNN, depending on ``cfg['model']['type']``."""
    raise NotImplementedError
