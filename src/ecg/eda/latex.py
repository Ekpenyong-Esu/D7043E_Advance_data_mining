"""Turn a pandas table into a LaTeX table in the report's style.

The output uses the macros defined in ``report/report.tex`` (``\\hd``, ``rowtint``, column types ``Y`` and ``R``).
"""
import numpy as np
import pandas as pd


def _cell(v, decimals: int) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}".replace(",", r"\,")
    if isinstance(v, (float, np.floating)):
        return f"{v:.{decimals}f}"
    return str(v)


def latex_table(df: pd.DataFrame, caption: str, label: str, align: str | None = None, decimals: int = 3) -> str:
    """Return a full ``table`` environment. The index is ignored, so reset it first if you want it shown.

    Integers get a thin space as thousands separator, floats ``decimals`` digits.
    """
    align = align or "Y" + "R" * (df.shape[1] - 1)
    head = " & ".join(rf"\hd{{{c}}}" for c in df.columns)
    body = [" & ".join(_cell(v, decimals) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines = [
        r"\begin{table}[H]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"  \rowcolors{2}{white}{rowtint}",
        rf"  \begin{{tabularx}}{{\linewidth}}{{{align}}}",
        r"    \rowcolor{primaryblue}",
        f"    {head} \\\\",
        *(f"    {b}" for b in body),
        r"    \bottomrule",
        r"  \end{tabularx}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"
