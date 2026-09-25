"""Save and load model weights, together with the config and score that produced them."""
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, cfg: dict, epoch: int, metric: float) -> None:
    """Write the weights plus the config, epoch number and validation score to one file."""
    torch.save({"state_dict": model.state_dict(), "config": cfg, "epoch": epoch, "metric": metric}, path)


def load_checkpoint(path: str | Path, model):
    """Load the weights into ``model``. Return the model and the saved dict (config, epoch, metric)."""
    ckpt = torch.load(path, map_location="cpu")  # works even if the file was saved on a GPU
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
