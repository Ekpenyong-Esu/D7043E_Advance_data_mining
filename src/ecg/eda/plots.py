"""The figures for report section 2.3. Each function draws one figure and returns it.

The caller decides where to save it. Colours come from ``style.py``.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecg.constants import CLASS_NAMES, FS_HZ, SYMBOL_TO_CLASS
from ecg.data.channels import select_lead
from ecg.data.records import Record
from ecg.eda.style import CLASS_COLORS, INK_SOFT


def raw_signal(r: Record, seconds: float = 10.0, start_s: float = 0.0):
    """Both channels of one record for a few seconds, with the annotated R-peaks marked on the top one."""
    a, b = int(start_s * r.fs), int((start_s + seconds) * r.fs)  # sample range to show
    t = np.arange(a, b) / r.fs                                    # time axis in seconds
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.4), sharex=True)
    for ax, ch in zip(axes, range(2)):
        ax.plot(t, r.signal[a:b, ch], color=INK_SOFT if ch else CLASS_COLORS["N"])
        ax.set_ylabel(f"{r.sig_names[ch]} (mV)")
    # Mark the annotated beats that fall inside the shown range.
    peaks = r.ann_samples[(r.ann_samples >= a) & (r.ann_samples < b)]
    axes[0].scatter(peaks / r.fs, r.signal[peaks, 0], s=14, color=CLASS_COLORS["S"], zorder=3, label="annotated R-peak")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("time (s)")
    fig.suptitle(f"Record {r.record_id}: {seconds:.0f} s of raw ECG", x=0.02, ha="left")
    return fig


def class_examples(recs: dict[int, Record], lead: str, before: int, after: int, n_per_class: int = 15):
    """One panel per class. Several beats are drawn on top of each other, lined up on the R-peak."""
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)
    t = np.arange(-before, after) / FS_HZ * 1000  # time axis in ms, 0 at the R-peak
    for ax, cls in zip(axes, CLASS_NAMES):
        drawn = 0
        # Walk through the records and beats until we have enough examples of this class.
        for r in recs.values():
            if lead not in r.sig_names:
                continue
            x = select_lead(r, lead)
            for p, s in zip(r.ann_samples, r.ann_symbols):
                if SYMBOL_TO_CLASS.get(s) != cls or p - before < 0 or p + after > len(x):
                    continue
                w = x[p - before:p + after]; w = (w - w.mean()) / (w.std() + 1e-8)  # z-score, like the pipeline
                ax.plot(t, w, color=CLASS_COLORS[cls], alpha=0.35, linewidth=0.8)
                drawn += 1
                if drawn >= n_per_class:
                    break
            if drawn >= n_per_class:
                break
        ax.set_title(cls, color=CLASS_COLORS[cls]); ax.set_xlabel("ms from R-peak")
        ax.axvline(0, color=INK_SOFT, linewidth=0.6, linestyle=":")  # dotted line at the R-peak
    axes[0].set_ylabel("normalised amplitude")
    return fig


def class_distribution(totals: pd.DataFrame):
    """Bar chart of beats per class, DS1 next to DS2. The y-axis is log scale because N dominates."""
    counts = totals["count"].loc[["DS1", "DS2"]]
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    x = np.arange(len(CLASS_NAMES)); w = 0.38  # bar positions and width
    for i, (split, hatch) in enumerate([("DS1", None), ("DS2", "////")]):  # DS2 bars are hatched
        vals = counts.loc[split, CLASS_NAMES].to_numpy(dtype=float)
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=[CLASS_COLORS[c] for c in CLASS_NAMES],
                      alpha=1.0 if hatch is None else 0.55, hatch=hatch, edgecolor="white", linewidth=0.8, label=split)
        for b, v in zip(bars, vals):  # write the count above each bar
            ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{int(v):,}", ha="center", va="bottom", fontsize=7, color=INK_SOFT)
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylabel("beats (log scale)"); ax.legend(loc="upper right")
    ax.set_ylim(top=float(counts.to_numpy().max()) * 4)  # leave room for the labels
    return fig


def record_distribution(cc: pd.DataFrame):
    """Stacked bar per record showing its beats per class. DS1 records on the left, DS2 on the right."""
    order = cc.sort_values("split").index  # DS1 first, then DS2
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    bottom = np.zeros(len(order))  # where the next class's bar segment starts
    for cls in CLASS_NAMES:
        vals = cc.loc[order, cls].values
        ax.bar(range(len(order)), vals, bottom=bottom, color=CLASS_COLORS[cls], label=cls, width=0.8, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=6)
    # Dashed line and labels to separate DS1 from DS2.
    split_change = list(cc.loc[order, "split"].values).index("DS2")
    ax.axvline(split_change - 0.5, color=INK_SOFT, linewidth=0.8, linestyle="--")
    ax.text(split_change / 2, bottom.max() * 1.02, "DS1", ha="center", fontsize=8, color=INK_SOFT)
    ax.text((split_change + len(order)) / 2, bottom.max() * 1.02, "DS2", ha="center", fontsize=8, color=INK_SOFT)
    ax.set_ylabel("beats"); ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    return fig


def amplitude_distribution(amp: pd.DataFrame):
    """Left: median R-peak height per record. Right: histogram of those values across records."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    axes[0].bar(range(len(amp)), amp["r_peak_median"], color=CLASS_COLORS["N"], width=0.8)
    axes[0].set_xticks(range(len(amp))); axes[0].set_xticklabels(amp.index, rotation=90, fontsize=6)
    axes[0].set_ylabel("median R-peak amplitude (mV)")
    axes[1].hist(amp["r_peak_median"], bins=15, color=CLASS_COLORS["N"], edgecolor="white")
    axes[1].set_xlabel("median R-peak amplitude (mV)"); axes[1].set_ylabel("records")
    return fig


def noisy_examples(noisy_recs: list[Record], noise: dict[str, np.ndarray], lead: str, fs: int, seconds: float = 5.0):
    """A few seconds of noisy MIT-BIH records on top, and the pure NSTDB noise types below."""
    n = len(noisy_recs) + len(noise)
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 1.1 * n), sharex=True)
    t = np.arange(int(seconds * fs)) / fs
    # Noisy records: start just before the first noise mark, so the noise is visible.
    for ax, r in zip(axes, noisy_recs):
        start = _first_noise_mark(r) or 0
        ax.plot(t, select_lead(r, lead)[start:start + len(t)], color=CLASS_COLORS["N"], linewidth=0.8)
        ax.set_ylabel(f"rec {r.record_id}", rotation=0, ha="right", va="center")
    # Pure noise signals from the NSTDB.
    for ax, (name, x) in zip(axes[len(noisy_recs):], noise.items()):
        ax.plot(t, x[:len(t)], color=CLASS_COLORS["S"], linewidth=0.8)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
    axes[-1].set_xlabel("time (s)")
    return fig


def _first_noise_mark(r: Record) -> int | None:
    """Sample index one second before the first ``~`` (noise) annotation, or None if there is none."""
    for p, s in zip(r.ann_samples, r.ann_symbols):
        if s == "~":
            return max(0, int(p) - r.fs)
    return None


def amplitude_scale_example(beat: np.ndarray, fs: int, before: int, factors=(1.0, 0.8, 1.2)):
    """One beat drawn at several amplitude scales, to show the augmentation for report section 3.7."""
    t = (np.arange(len(beat)) - before) / fs * 1000  # ms from the R-peak
    colors = [CLASS_COLORS["N"], CLASS_COLORS["V"], CLASS_COLORS["S"]]
    fig, axes = plt.subplots(1, len(factors), figsize=(3 * len(factors), 2.8), sharey=True)
    for ax, k, col in zip(axes, factors, colors):
        ax.plot(t, beat, color=INK_SOFT, linewidth=0.8, linestyle="--")  # original, dashed, for reference
        ax.plot(t, beat * k, color=col, linewidth=1.6)                   # scaled version
        ax.axhline(0, color=INK_SOFT, linewidth=0.5)
        peak = float(beat.max() * k)
        ax.annotate(f"peak = {peak:.1f}", xy=(0, peak), xytext=(60, peak), fontsize=8, color=col, va="center")
        ax.set_title("original" if k == 1.0 else f"scaled  x {k}"); ax.set_xlabel("ms from R-peak")
    axes[0].set_ylabel("normalised amplitude")
    return fig
