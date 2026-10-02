"""Save the processed splits, and write the summary tables for results/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ecg.constants import CLASS_NAMES, EXCLUDED_BEAT_SYMBOLS, SYMBOL_TO_CLASS


def save_split(path: str | Path, X: np.ndarray, y: np.ndarray, record_ids: np.ndarray, rr: np.ndarray) -> None:
    """Write one split to a compressed .npz file with four arrays: X, y, record_ids, rr."""
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64), record_ids=record_ids, rr=rr.astype(np.float32))


def split_statistics(counts: dict[str, np.ndarray], records: dict[str, list[int]]) -> pd.DataFrame:
    """Build the table for report section 3.8: one row per split with the beat count per class.

    ``counts`` maps a split name to its label array. ``records`` maps it to its record list.
    """
    rows = []
    for name, y in counts.items():
        per_class = {c: int((y == i).sum()) for i, c in enumerate(CLASS_NAMES)}
        rows.append({"split": name, "records": len(records[name]), **per_class, "total": int(len(y))})
    return pd.DataFrame(rows)


def record_statistics(per_record: list[dict]) -> pd.DataFrame:
    """Build the per-record table: split, class counts, edge-dropped and excluded counts.

    Records without excluded beats have no ``excluded_*`` key, so those cells become 0.
    """
    return pd.DataFrame(per_record).fillna(0).astype({c: int for c in CLASS_NAMES})


def write_label_mapping(path: str | Path) -> None:
    """Write the class names, symbol mapping and excluded symbols to a JSON file, for the record."""
    payload = {
        "classes": CLASS_NAMES,
        "symbol_to_class": SYMBOL_TO_CLASS,
        "excluded_beat_symbols": sorted(EXCLUDED_BEAT_SYMBOLS),
    }
    Path(path).write_text(json.dumps(payload, indent=2))
