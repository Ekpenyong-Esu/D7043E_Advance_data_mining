"""The same CNN as ``cnn.py``, built with ai8x layer wrappers. Not written yet.

The ai8x layers come from the ai8x-training repo. They are needed for
quantisation-aware training and for ``ai8xize.py`` to synthesise the model.
This module only imports when ai8x is installed.
"""
try:
    import ai8x
except ImportError:  # pragma: no cover
    ai8x = None


def build_ai8x_cnn(channels: list[int], kernels: list[int], pools: list[int], num_classes: int = 4, **kwargs):
    """Build the network. Raises a clear error if ai8x-training is not installed."""
    if ai8x is None:
        raise ImportError("ai8x-training is not installed; see STEP_BY_STEP.md Step 9.1")
    raise NotImplementedError
