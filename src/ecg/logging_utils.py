"""One helper that gives every module a logger with the same format."""
import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints lines like ``2026-09-24 10:00:00 INFO ecg.data.pipeline: ...``."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # only set up once, even if called many times
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
