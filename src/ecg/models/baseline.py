"""Simple classical models to compare the CNN against (report section 4.1).

They see the flattened beat (256 values) plus the three RR-interval features. The
CNN sees only the beat. The RR features are the standard hand-made timing features
that tell an early S beat from an N beat, so the baseline is a fair, strong reference.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def baseline_features(X: np.ndarray, rr: np.ndarray) -> np.ndarray:
    """Concatenate the flattened beat (n, L) with the RR features (n, 3)."""
    return np.concatenate([X.reshape(len(X), -1), rr], axis=1).astype(np.float32)


def build_baseline(model_cfg: dict, seed: int) -> Pipeline:
    """Return an untrained sklearn pipeline chosen by ``model_cfg['type']``.

    ``class_weight="balanced"`` makes mistakes on rare classes cost more, the same
    idea as the weighted loss of the CNN.
    """
    kind = model_cfg["type"]
    if kind == "random_forest":
        clf = RandomForestClassifier(n_estimators=int(model_cfg.get("n_estimators", 200)),
                                     class_weight=model_cfg.get("class_weight", "balanced"),
                                     min_samples_leaf=int(model_cfg.get("min_samples_leaf", 1)),
                                     n_jobs=-1, random_state=seed)
        return make_pipeline(clf)
    if kind == "logistic_regression":
        clf = LogisticRegression(max_iter=int(model_cfg.get("max_iter", 2000)),
                                 C=float(model_cfg.get("C", 1.0)),
                                 class_weight=model_cfg.get("class_weight", "balanced"))
        return make_pipeline(StandardScaler(), clf)
    raise ValueError(f"Unknown baseline type: {kind}")


def n_parameters(pipe: Pipeline) -> int:
    """A size number comparable to the CNN's weight count: coefficients for LR, tree nodes for RF."""
    clf = pipe.steps[-1][1]
    if hasattr(clf, "coef_"):
        return int(clf.coef_.size + clf.intercept_.size)
    return int(sum(t.tree_.node_count for t in clf.estimators_))
