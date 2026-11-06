"""One function that turns the ``model:`` block of a config into a model object."""
from ecg.models.cnn import ECGCNN

BASELINES = {"random_forest", "logistic_regression"}


def build_cnn(cfg: dict, ai8x=None, input_length: int = 256):
    """Build the CNN from a model config.

    With ``ai8x`` (the imported ai8x module, see ``ecg.deployment.toolchain``) the network
    uses the ai8x wrappers, which is required for quantisation and synthesis. Without it,
    the plain PyTorch version is built; it has the same layers but cannot be quantised.
    """
    m = cfg["model"]
    if m["type"] != "cnn1d":
        raise ValueError(f"Not a CNN config: {m['type']}")
    if ai8x is not None:
        from ecg.models.ai8x_cnn import build_ai8x_cnn  # imported here: it needs the toolchain

        return build_ai8x_cnn(ai8x, m, input_length)
    return ECGCNN(input_length=input_length, **{k: v for k, v in m.items() if k != "type"})
