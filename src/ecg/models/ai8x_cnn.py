"""Same architecture as ``cnn.py`` expressed with ai8x layer wrappers.

Only imported when ``ai8x`` (from ai8x-training) is on the path. Keep the
layer list identical to ``ECGCNN`` so FP32 and QAT/synthesis runs match.
"""
try:
    import ai8x  # type: ignore
except ImportError:  # pragma: no cover
    ai8x = None


def build_ai8x_cnn(channels: list[int], kernels: list[int], pools: list[int], num_classes: int = 4, **kwargs):
    if ai8x is None:
        raise ImportError("ai8x-training is not installed; see STEP_BY_STEP.md Step 9.1")
    raise NotImplementedError
