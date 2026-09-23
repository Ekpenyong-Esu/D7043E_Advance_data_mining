"""Classical baselines on flattened beats (report §4.1)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_baseline(cfg: dict):
    kind = cfg["type"]
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 200),
            class_weight=cfg.get("class_weight", "balanced"),
            n_jobs=-1,
            random_state=0,
        )
    if kind == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight=cfg.get("class_weight", "balanced"))
    raise ValueError(f"Unknown baseline type: {kind}")
