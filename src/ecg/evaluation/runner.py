"""Score one checkpoint on one split and write all the evaluation files. Not written yet."""


def run_evaluation(cfg: dict, split: str, checkpoint: str | None = None) -> dict:
    """Load the model and the split, predict, then write metrics, confusion matrix and error examples."""
    raise NotImplementedError
