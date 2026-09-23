"""Load checkpoint + split, predict, write metrics / confusion / errors."""


def run_evaluation(cfg: dict, split: str, checkpoint: str | None = None) -> dict:
    raise NotImplementedError
