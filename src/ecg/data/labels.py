"""Turn MIT-BIH annotation symbols into class indices 0..3, and count what was dropped."""
import numpy as np

from ecg.constants import CLASS_TO_INDEX, EXCLUDED_BEAT_SYMBOLS, SYMBOL_TO_CLASS


def map_symbols(symbols: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Map each symbol to a class index.

    Returns three things:
      labels    one number per symbol. Only valid where ``keep`` is True.
      keep      True for real beats we use, False for everything else.
      excluded  how many beats of each excluded symbol were dropped, for the report.
    """
    labels = np.full(len(symbols), -1, dtype=np.int64)  # -1 means "not a beat we use"
    keep = np.zeros(len(symbols), dtype=bool)
    excluded: dict[str, int] = {}
    for i, sym in enumerate(symbols):
        cls = SYMBOL_TO_CLASS.get(sym)
        if cls is not None:                       # a beat in one of the four classes
            labels[i] = CLASS_TO_INDEX[cls]
            keep[i] = True
        elif sym in EXCLUDED_BEAT_SYMBOLS:        # a beat we drop on purpose, count it
            excluded[sym] = excluded.get(sym, 0) + 1
        # anything else is not a heartbeat (rhythm change, noise mark, ...) and is skipped
    return labels, keep, excluded
