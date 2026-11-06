"""Save and load model weights together with the config and score that produced them.

The file layout is the one the ai8x toolchain expects (``state_dict``, ``arch``,
``epoch``, ``extras``), so the same file goes straight into ``quantize.py``.
"""
from pathlib import Path

import torch


def save_checkpoint(path: str | Path, model, cfg: dict, epoch: int, metric: float) -> None:
    """Write the weights plus the config, epoch number and validation score to one file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": cfg.get("arch", cfg["name"]), "config": cfg,
                "epoch": epoch, "metric": metric, "extras": {}}, path)


def has_batchnorm(state_dict: dict) -> bool:
    """True when the weights still contain BatchNorm layers (not yet folded into the convolutions)."""
    return any(".bn." in k for k in state_dict)


def load_checkpoint(path: str | Path, model, ai8x=None):
    """Load the weights into ``model``. Returns ``(model, saved dict)``.

    A checkpoint saved after BatchNorm folding (QAT or ``fuse_checkpoint``) has no ``bn`` weights;
    the model is then folded too before loading, which needs the ``ai8x`` module.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)  # the dict holds the config too
    if ai8x is not None and not has_batchnorm(ckpt["state_dict"]) and has_batchnorm(model.state_dict()):
        ai8x.fuse_bn_layers(model)
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt


def fuse_checkpoint(cfg: dict, ai8x, src: str | Path, dst: str | Path) -> Path:
    """Fold the BatchNorm layers of an FP32 checkpoint into the conv weights and save the result.

    The accelerator has no BatchNorm; ``quantize.py`` expects the folded weights. Returns ``dst``.
    """
    from ecg.models.registry import build_cnn  # imported here to keep this module light

    model = build_cnn(cfg, ai8x, cfg["input_length"])
    _, ckpt = load_checkpoint(src, model)
    ai8x.fuse_bn_layers(model)
    save_checkpoint(dst, model, cfg, ckpt["epoch"], ckpt["metric"])
    return Path(dst)
