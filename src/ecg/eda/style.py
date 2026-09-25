"""Colours and matplotlib settings shared by every figure, so they all look the same."""
import matplotlib as mpl

from ecg.constants import CLASS_NAMES

# One colour per class, always in the order N, S, V, F. Chosen to be readable
# for people with colour-vision deficiency.
CLASS_COLORS = dict(zip(CLASS_NAMES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK = "#0b0b0b"        # main text
INK_SOFT = "#52514e"   # axis labels, tick labels, secondary lines
GRID = "#e6e5e1"       # grid lines and axis edges
SURFACE = "#ffffff"    # figure background


def apply_style() -> None:
    """Set matplotlib defaults. Call once before drawing any figure."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,   # hide the top and right box lines
        "xtick.color": INK_SOFT, "ytick.color": INK_SOFT, "text.color": INK,
        "axes.labelcolor": INK_SOFT, "lines.linewidth": 1.2,
        "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
