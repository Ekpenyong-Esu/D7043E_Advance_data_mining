"""Save / load model weights together with the config that produced them."""
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, cfg: dict, epoch: int, metric: float) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "metric": metric}, path)


def load_checkpoint(path: str | Path, model):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
