"""Turn a saved model into one function ``predict(X, rr) -> class indices``.

Every evaluation (clean DS2, noise test, error analysis) goes through these functions,
so the FP32 CNN, the INT8 CNN and the baselines are scored in exactly the same way.
"""
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np

from ecg.datasets.beat_dataset import to_model_input
from ecg.deployment.toolchain import MAX78002, import_ai8x
from ecg.models.baseline import baseline_features
from ecg.models.registry import build_cnn
from ecg.training.checkpoint import load_checkpoint
from ecg.training.loop import pick_device, predict_array

Predictor = Callable[[np.ndarray, np.ndarray], np.ndarray]


def cnn_predictor(cfg: dict, checkpoint: str | Path, int8: bool = False) -> Predictor:
    """Load a CNN checkpoint. ``int8=True`` loads a ``quantize.py`` output and runs it in INT8 simulation.

    The ai8x layers read the device mode from a global, so the mode is set again on every call.
    """
    ai8x = import_ai8x(cfg, simulate=int8)
    device = pick_device(cfg["training"].get("device", "auto"))
    model = build_cnn(cfg, ai8x, cfg["input_length"])
    load_checkpoint(checkpoint, model, ai8x)
    ai8x.update_model(model)  # refresh the quantisation functions after loading
    model.to(device).eval()

    def predict(X: np.ndarray, rr: np.ndarray | None = None) -> np.ndarray:
        ai8x.set_device(MAX78002, int8, False, verbose=False)
        out = predict_array(model, to_model_input(X, cfg["input_scale"], int8), device).argmax(1)
        ai8x.set_device(MAX78002, False, False, verbose=False)  # back to FP32 behaviour for everyone else
        return out

    return predict


def cnn_scores(cfg: dict, checkpoint: str | Path, X: np.ndarray, int8: bool = False) -> np.ndarray:
    """Raw output scores (n, 4) of a CNN checkpoint, used to compare Python with the toolchain."""
    ai8x = import_ai8x(cfg, simulate=int8)
    device = pick_device(cfg["training"].get("device", "auto"))
    model = build_cnn(cfg, ai8x, cfg["input_length"])
    load_checkpoint(checkpoint, model, ai8x)
    ai8x.update_model(model)
    out = predict_array(model.to(device).eval(), to_model_input(X, cfg["input_scale"], int8), device)
    ai8x.set_device(MAX78002, False, False, verbose=False)
    return out


def baseline_predictor(path: str | Path) -> Predictor:
    """Load a saved sklearn pipeline (``joblib``) and wrap it."""
    pipe = joblib.load(path)

    def predict(X: np.ndarray, rr: np.ndarray) -> np.ndarray:
        return pipe.predict(baseline_features(X, rr))

    return predict
