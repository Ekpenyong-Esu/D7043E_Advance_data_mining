"""Summary tables for the Data Understanding section of the report.

Each function takes the loaded records and returns a pandas table.
Nothing here reads or writes files except ``load_records``.
"""
import numpy as np
import pandas as pd

from ecg.constants import CLASS_NAMES, DS1, DS2, SYMBOL_TO_CLASS
from ecg.data.labels import map_symbols
from ecg.data.records import Record, read_record


def load_records(raw_dir: str, record_ids: list[int]) -> dict[int, Record]:
    """Read every record in ``record_ids``. Returns a dict: record id -> Record."""
    return {rid: read_record(raw_dir, rid) for rid in record_ids}


def symbol_counts(recs: dict[int, Record]) -> pd.DataFrame:
    """Count every annotation symbol per record, exactly as it appears in the .atr file."""
    rows = []
    for rid, r in recs.items():
        syms, counts = np.unique(r.ann_symbols, return_counts=True)
        rows.append({"record": rid, **dict(zip(syms, counts.tolist()))})
    # A symbol missing from a record becomes NaN, so fill with 0.
    return pd.DataFrame(rows).fillna(0).astype(int).set_index("record")


def class_counts(recs: dict[int, Record]) -> pd.DataFrame:
    """Count N/S/V/F beats per record, using the same mapping as the pipeline. Also notes the split."""
    rows = []
    for rid, r in recs.items():
        y, keep, excluded = map_symbols(r.ann_symbols)
        counts = {c: int((y[keep] == i).sum()) for i, c in enumerate(CLASS_NAMES)}
        split = "DS1" if rid in DS1 else "DS2" if rid in DS2 else "excluded"
        rows.append({"record": rid, "split": split, **counts, "excluded": sum(excluded.values())})
    return pd.DataFrame(rows).set_index("record")


def class_totals(cc: pd.DataFrame) -> pd.DataFrame:
    """Sum the per-record counts into DS1, DS2 and All, with counts and percentages side by side."""
    by = cc.groupby("split")[CLASS_NAMES].sum()
    by.loc["All"] = by.sum()
    pct = by.div(by.sum(axis=1), axis=0) * 100  # each row sums to 100
    return pd.concat({"count": by, "percent": pct.round(2)}, axis=1)


def lead_table(recs: dict[int, Record]) -> pd.DataFrame:
    """One row per record: the two channel names, the sampling rate and the number of samples."""
    return pd.DataFrame({rid: {"ch0": r.sig_names[0], "ch1": r.sig_names[1], "fs": r.fs,
                               "samples": r.signal.shape[0]} for rid, r in recs.items()}).T


def amplitude_stats(recs: dict[int, Record], lead: str) -> pd.DataFrame:
    """Per record: min, max and spread of the chosen lead, and the median R-peak height.

    Records that do not have the lead are skipped.
    """
    rows = []
    for rid, r in recs.items():
        if lead not in r.sig_names:
            continue
        x = r.signal[:, r.sig_names.index(lead)]
        beat_mask = np.isin(r.ann_symbols, list(SYMBOL_TO_CLASS))  # only real beats, not noise marks
        peaks = r.ann_samples[beat_mask]
        rows.append({"record": rid, "min": x.min(), "max": x.max(), "std": x.std(),
                     "r_peak_median": np.median(x[peaks]), "nan": int(np.isnan(x).sum())})
    return pd.DataFrame(rows).set_index("record")


def noise_annotation_counts(recs: dict[int, Record]) -> pd.Series:
    """How many ``~`` (noise) marks each record has. More marks means a noisier record."""
    return pd.Series({rid: r.ann_symbols.count("~") for rid, r in recs.items()}, name="noise_marks")
