"""One function that turns the ``model:`` block of a config into a model object."""
from ecg.models.baseline import build_baseline
from ecg.models.cnn import ECGCNN


def build_model(model_cfg: dict, prefer_ai8x: bool = False):
    """Build the model named in ``model_cfg['type']``.

    ``prefer_ai8x=True`` builds the CNN with ai8x layers instead of plain PyTorch.
    That version is needed for quantisation-aware training and for synthesis.
    """
    kind = model_cfg["type"]
    if kind in {"random_forest", "logistic_regression"}:
        return build_baseline(model_cfg)
    if kind == "cnn1d":
        if prefer_ai8x:
            from ecg.models.ai8x_cnn import build_ai8x_cnn  # imported here because ai8x may not be installed

            return build_ai8x_cnn(**{k: v for k, v in model_cfg.items() if k != "type"})
        return ECGCNN(model_cfg["channels"], model_cfg["kernels"], model_cfg["pools"], model_cfg["num_classes"])
    raise ValueError(f"Unknown model type: {kind}")
