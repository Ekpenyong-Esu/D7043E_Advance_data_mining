"""Wires data, model, loss, loop and logging for one training run."""
from ecg.logging_utils import get_logger

log = get_logger(__name__)


def run_training(cfg: dict) -> None:
    """Dispatch: sklearn baseline or torch CNN, based on cfg['model']['type']."""
    raise NotImplementedError
