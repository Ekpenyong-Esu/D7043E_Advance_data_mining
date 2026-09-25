"""Entry point: write the numbers from results/ into the report text.

The Submission 1 sections contain ``\\tbd`` placeholders. This script replaces
them with the real numbers from ``results/split_stats.csv`` and
``results/eda/class_totals.csv``, plus the installed package versions.
It is safe to run twice: a placeholder that is already filled is left alone.
"""
import importlib.metadata as md
import re
from pathlib import Path

import pandas as pd

from ecg.constants import CLASS_NAMES

SEC = Path("report/sections")


def _fmt(n) -> str:
    """Format 45866 as ``45\\,866`` (LaTeX thin space as the thousands separator)."""
    return f"{int(n):,}".replace(",", "\\,")


def fill_class_table(text: str, totals: pd.DataFrame) -> str:
    """Section 2.2: total beats per class over all records."""
    for c in CLASS_NAMES:
        text = text.replace(f"\\tbd[{c}] \\\\", f"{_fmt(totals.loc['All', ('count', c)])} \\\\")
    return text


def fill_final_dataset(text: str, stats: pd.DataFrame) -> str:
    """Section 3.8: the train / validation / test table."""
    rows = {"train": "Train", "val": "Validation", "test": "Test (DS2)"}
    for key, label in rows.items():
        r = stats[stats.split == key].iloc[0]
        cells = " & ".join(_fmt(r[c]) for c in CLASS_NAMES + ["total"])
        # Match "Label & <records> & \tbd & \tbd & \tbd & \tbd & \tbd \\" and keep the record count.
        pat = re.compile(re.escape(label) + r"\s*&\s*(\d+)\s*& (\\tbd & ){4}\\tbd \\\\")
        text = pat.sub(lambda m: f"{label} & {m.group(1)} & {cells} \\\\", text)
    return text


def fill_imbalance(text: str, stats: pd.DataFrame) -> str:
    """Section 3.6: class percentages in the training set and the largest-to-smallest ratio."""
    r = stats[stats.split == "train"].iloc[0]
    pct = {c: 100 * r[c] / r["total"] for c in CLASS_NAMES}
    for c in CLASS_NAMES:
        text = text.replace(f"\\tbd[{c}\\,\\%]", f"{pct[c]:.1f}\\,\\%")
    ratio = r[CLASS_NAMES].max() / r[CLASS_NAMES].min()
    return text.replace("\\tbd[ratio]", f"{ratio:.0f}")


def fill_versions(text: str) -> str:
    """Section 3.9: installed versions of the main packages."""
    for pkg, name in [("wfdb", "wfdb"), ("numpy", "NumPy"), ("scipy", "SciPy"), ("torch", "PyTorch"), ("scikit-learn", "scikit-learn")]:
        try:
            ver = md.version(pkg)
        except md.PackageNotFoundError:
            continue  # package not installed, leave the placeholder
        text = text.replace(f"{name} \\tbd[ver]", f"{name} {ver}")
    return text


def main() -> None:
    stats = pd.read_csv("results/split_stats.csv")
    totals = pd.read_csv("results/eda/class_totals.csv", header=[0, 1], index_col=0)  # two header rows
    f02 = SEC / "02_data_understanding.tex"; f03 = SEC / "03_data_preparation.tex"
    f02.write_text(fill_class_table(f02.read_text(), totals))
    t = f03.read_text()
    t = fill_final_dataset(t, stats); t = fill_imbalance(t, stats); t = fill_versions(t)
    f03.write_text(t)
    remaining = sum(f.read_text().count("\\tbd") for f in [f02, f03])
    print(f"filled; remaining placeholders in 02/03: {remaining}")


if __name__ == "__main__":
    main()
