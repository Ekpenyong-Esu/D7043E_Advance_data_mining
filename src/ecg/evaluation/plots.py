"""The figures for report sections 4 to 6. Each function draws one figure and returns it.

The caller decides where to save it. Colours come from ``ecg.eda.style``.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg.constants import CLASS_NAMES
from ecg.eda.style import CLASS_COLORS, INK_SOFT

SNR_ORDER = ["clean", "12", "6", "0"]


def learning_curve(history: pd.DataFrame, title: str = ""):
    """Training loss and validation macro-F1 per epoch, with the chosen epoch marked."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax[0].plot(history["epoch"], history["loss"], color=CLASS_COLORS["N"]); ax[0].set_ylabel("training loss")
    ax[1].plot(history["epoch"], history["val_macro_f1"], color=CLASS_COLORS["V"], label="macro-F1")
    for c in ("S", "F"):
        ax[1].plot(history["epoch"], history[f"val_recall_{c}"], color=CLASS_COLORS[c], linewidth=0.9, label=f"recall {c}")
    best = history["val_macro_f1"].idxmax()
    ax[1].axvline(history.loc[best, "epoch"], color=INK_SOFT, linestyle=":", linewidth=0.8)
    ax[1].set_ylabel("validation score"); ax[1].set_ylim(0, 1); ax[1].legend(loc="lower right", fontsize=7)
    for a in ax:
        a.set_xlabel("epoch")
    if title:
        fig.suptitle(title, x=0.02, ha="left")
    return fig


def confusion_matrix(cm: np.ndarray, title: str = ""):
    """Counts in each cell, shaded by the share of the true class (each row sums to one)."""
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]:,}\n{share[i, j]:.1%}", ha="center", va="center", fontsize=7,
                    color="white" if share[i, j] > 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASS_NAMES); ax.set_yticks(range(4)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.grid(False)
    if title:
        ax.set_title(title, fontsize=9, loc="left")
    return fig


def noise_curve(df: pd.DataFrame, model: str):
    """Left: macro-F1 against SNR, one line per noise type. Right: per-class F1 averaged over noise types."""
    d = df[df["model"] == model].copy(); d["snr"] = d["snr_db"].astype(str)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for kind, col in zip(("bw", "em", "ma"), (CLASS_COLORS["N"], CLASS_COLORS["S"], CLASS_COLORS["V"])):
        s = d[(d["noise"] == kind) | (d["noise"] == "none")].set_index("snr").reindex(SNR_ORDER)
        ax[0].plot(SNR_ORDER, s["macro_f1"], marker="o", color=col, label=kind)
    ax[0].set_ylabel("macro-F1 on DS2"); ax[0].set_ylim(0, 1); ax[0].legend(loc="lower left", fontsize=7)
    mean = d.groupby("snr")[[f"f1_{c}" for c in CLASS_NAMES]].mean().reindex(SNR_ORDER)
    for c in CLASS_NAMES:
        ax[1].plot(SNR_ORDER, mean[f"f1_{c}"], marker="o", color=CLASS_COLORS[c], label=c)
    ax[1].set_ylabel("per-class F1 (mean over noise types)"); ax[1].set_ylim(0, 1); ax[1].legend(loc="lower left", fontsize=7, ncol=2)
    for a in ax:
        a.set_xlabel("SNR (dB)")
    return fig


def error_examples(X: np.ndarray, picks: dict, fs: int, before: int, per_row: int = 4):
    """One row per (true, predicted) pair, a few misclassified beats per row, aligned on the R-peak."""
    rows = [(pair, idx) for pair, idx in picks.items() if len(idx)]
    fig, axes = plt.subplots(len(rows), per_row, figsize=(7.0, 1.5 * len(rows)), sharex=True, sharey=True, squeeze=False)
    t = (np.arange(X.shape[-1]) - before) / fs * 1000
    step = 300 if t.max() - t.min() > 1000 else 200   # tick spacing that suits the window length
    ticks = np.arange(np.ceil(t.min() / step) * step, t.max() + 1, step)
    for r, ((true, pred), idx) in enumerate(rows):
        for c in range(per_row):
            ax = axes[r, c]
            if c < len(idx):
                ax.plot(t, X[idx[c], 0], color=CLASS_COLORS[true], linewidth=0.9)
                ax.axvline(0, color=INK_SOFT, linewidth=0.5, linestyle=":")
            ax.set_xticks(ticks); ax.tick_params(labelsize=6)
        axes[r, 0].set_ylabel(f"true {true}\npred {pred}", rotation=0, ha="right", va="center", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ms from R-peak", fontsize=7)
    fig.tight_layout()
    return fig


def fp32_vs_int8(df: pd.DataFrame, models: list[str]):
    """Bars of macro-F1 and per-class recall for the models in ``models`` (rows of results/metrics.csv)."""
    d = df.set_index("model").loc[models]
    cols = ["macro_f1"] + [f"recall_{c}" for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    w = 0.8 / len(models); x = np.arange(len(cols))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, d.loc[m, cols], w, label=m, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(["macro-F1"] + [f"recall {c}" for c in CLASS_NAMES]); ax.set_ylim(0, 1)
    ax.legend(loc="lower left", fontsize=7, ncol=len(models))
    return fig
