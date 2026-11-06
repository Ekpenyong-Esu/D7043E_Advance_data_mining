"""Run one training job from start to end: the CNN (FP32 or QAT fine-tune) or a baseline.

Builds the loaders, model and loss, trains, checks the validation score after every
epoch, keeps the best checkpoint and writes one row to results/experiments.csv.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import torch

from ecg.datasets.loaders import build_loaders, load_split
from ecg.deployment.toolchain import MAX78002
from ecg.evaluation.metrics import flat_scores
from ecg.logging_utils import get_logger
from ecg.models.baseline import baseline_features, build_baseline, n_parameters
from ecg.models.cnn import count_parameters
from ecg.models.registry import build_cnn
from ecg.seed import set_seed
from ecg.training.checkpoint import load_checkpoint, save_checkpoint
from ecg.training.early_stopping import EarlyStopping
from ecg.training.experiment_log import log_run
from ecg.training.loop import pick_device, predict_loader, train_one_epoch
from ecg.training.losses import build_loss
from ecg.training.qat import prepare_qat

log = get_logger(__name__)


def fit(cfg: dict, model, loaders: dict, device, epochs: int, lr: float, ckpt_path: Path) -> tuple[pd.DataFrame, dict]:
    """The epoch loop. Returns the per-epoch history and the row of the best epoch."""
    tr = cfg["training"]
    loss_fn = build_loss(tr, loaders["train"].dataset.y, cfg["model"]["num_classes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=float(tr.get("weight_decay", 0.0)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if tr.get("scheduler") == "cosine" else None
    stopper = EarlyStopping(int(tr["early_stopping_patience"]))
    history, best = [], {}
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], loss_fn, optimizer, device)
        if scheduler:
            scheduler.step()
        y_true, y_pred = predict_loader(model, loaders["val"], device)
        row = {"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in flat_scores(y_true, y_pred).items()}}
        history.append(row)
        stop = stopper.step(row[f"val_{tr['selection_metric']}"])
        if stopper.improved:
            best = row
            save_checkpoint(ckpt_path, model, cfg, epoch, best[f"val_{tr['selection_metric']}"])
        log.info("epoch %3d loss %.4f val macro-F1 %.3f acc %.3f recall S %.2f F %.2f%s", epoch, loss, row["val_macro_f1"],
                 row["val_accuracy"], row["val_recall_S"], row["val_recall_F"], "  *" if stopper.improved else "")
        if stop:
            break
    return pd.DataFrame(history), best


def run_cnn_training(cfg: dict, ai8x=None, qat_from: str | Path | None = None) -> dict:
    """Train the CNN named in ``cfg``. With ``qat_from`` the FP32 weights are loaded from that
    checkpoint and quantisation-aware fine-tuning is run instead (needs ``ai8x``)."""
    set_seed(cfg["seed"])
    device = pick_device(cfg["training"].get("device", "auto"))
    loaders = build_loaders(cfg, cfg["paths"]["processed"])
    if ai8x is not None:
        ai8x.set_device(MAX78002, False, False, verbose=False)  # FP32 (or QAT) mode, never INT8 simulation
    model = build_cnn(cfg, ai8x, cfg["input_length"])
    tag, epochs, lr = "fp32", int(cfg["training"]["epochs"]), float(cfg["training"]["lr"])
    if qat_from is not None:
        load_checkpoint(qat_from, model)
        ai8x.fuse_bn_layers(model)   # the accelerator has no BatchNorm: fold it into the conv weights first
        policy = {"start_epoch": 0, "weight_bits": int(cfg["qat"]["weight_bits"]),
                  "activation_thresholds": bool(cfg["qat"].get("activation_thresholds", True)),
                  "outlier_removal_z_score": float(cfg["qat"].get("outlier_removal_z_score", 8.0))}
        layer_names = [f"conv{i}" for i in range(1, model.n_blocks + 1)] + ["fc"]
        prepare_qat(ai8x, model.to(device), loaders["train"], device, policy, layer_names)
        tag, epochs, lr = "qat", int(cfg["qat"]["epochs"]), float(cfg["qat"]["lr"])
    model.to(device)
    name = f"{cfg['name']}_{tag}"
    ckpt = Path(cfg["paths"]["checkpoints"]) / f"{name}.pth.tar"
    log.info("training %s on %s, %d parameters", name, device, count_parameters(model))
    history, best = fit(cfg, model, loaders, device, epochs, lr, ckpt)
    row = {"name": name, "model": "cnn1d", "n_params": count_parameters(model), "window": cfg["input_length"],
           "channels": str(cfg["model"]["channels"]), "batchnorm": cfg["model"].get("batchnorm", False),
           "dropout": cfg["model"].get("dropout", 0.0), "loss": cfg["training"]["loss"],
           "class_weight_power": cfg["training"].get("class_weight_power", 0) if cfg["training"]["loss"] != "ce" else 0,
           "oversample_power": cfg["training"].get("oversample_power", 0),
           "augmentation": cfg.get("augmentation", {}).get("enabled", True), "qat": tag == "qat",
           "epochs_run": len(history), "best_epoch": best.get("epoch"),
           **{k: v for k, v in best.items() if k.startswith("val_")}, "overrides": json.dumps(cfg.get("overrides", {}))}
    log_run(cfg["paths"]["experiments"], row)
    return {"name": name, "checkpoint": ckpt, "n_params": row["n_params"], "history": history, "best": row}


def run_baseline(cfg: dict) -> dict:
    """Train a baseline on the flattened beats plus RR features; score it on validation."""
    set_seed(cfg["seed"])
    tr, va = (load_split(cfg["paths"]["processed"], s) for s in ("train", "val"))
    pipe = build_baseline(cfg["model"], cfg["seed"]).fit(baseline_features(tr["X"], tr["rr"]), tr["y"])
    y_pred = pipe.predict(baseline_features(va["X"], va["rr"]))
    path = Path(cfg["paths"]["checkpoints"]) / f"{cfg['name']}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(pipe, path)
    row = {"name": cfg["name"], "model": cfg["model"]["type"], "n_params": n_parameters(pipe), "window": cfg["input_length"],
           **{f"val_{k}": v for k, v in flat_scores(va["y"], y_pred).items()}, "overrides": json.dumps(cfg.get("overrides", {}))}
    log_run(cfg["paths"]["experiments"], row)
    log.info("%s: val macro-F1 %.3f acc %.3f", cfg["name"], row["val_macro_f1"], row["val_accuracy"])
    return {"name": cfg["name"], "checkpoint": path, "n_params": row["n_params"], "best": row}
