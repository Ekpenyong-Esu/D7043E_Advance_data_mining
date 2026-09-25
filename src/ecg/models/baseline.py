"""Simple classical models to compare the CNN against (report section 4.1).

They work on the flattened beat (256 numbers) plus the RR features.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_baseline(cfg: dict):
    """Return an untrained sklearn model chosen by ``cfg['type']``.

    ``class_weight="balanced"`` makes the model pay more attention to the rare classes.
    """
    kind = cfg["type"]
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=-1,          # use all CPU cores
            random_state=0,
        )
    if kind == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=cfg.get("class_weight", "balanced"))
    raise ValueError(f"Unknown baseline type: {kind}")
