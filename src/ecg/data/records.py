"""Read one MIT-BIH record: the ECG signal and the beat annotations.

Every record in the database is stored as three files with the same name:
  NNN.dat  the ECG signal
  NNN.hea  the header (sampling rate, channel names, units)
  NNN.atr  the annotations (where each beat is and what type it is)
The wfdb library reads these files for us.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb


@dataclass
class Record:
    """Everything we need from one record, in plain Python and NumPy types."""

    record_id: int            # the record number, for example 101
    fs: int                   # sampling rate in Hz (360 for every MIT-BIH record)
    sig_names: list[str]      # channel names, for example ["MLII", "V1"]
    signal: np.ndarray        # the ECG, shape (n_samples, n_channels), in millivolts
    ann_samples: np.ndarray   # sample index of each annotation (for beats, the R-peak position)
    ann_symbols: list[str]    # one symbol per annotation, for example "N", "V", "A". "~" marks noise


def read_record(raw_dir: str | Path, record_id: int) -> Record:
    """Read the signal and annotations of one record from ``raw_dir``.

    ``raw_dir`` is the folder that holds the .dat, .hea and .atr files.
    ``record_id`` is the record number, for example 101.
    """
    # wfdb wants the path without the file extension, e.g. "data/raw/mitdb/101".
    path = str(Path(raw_dir) / str(record_id))

    rec = wfdb.rdrecord(path)     # reads .dat and .hea
    ann = wfdb.rdann(path, "atr")  # reads .atr

    return Record(
        record_id=record_id,
        fs=int(rec.fs),
        sig_names=list(rec.sig_name),
        signal=rec.p_signal,               # "p" means physical units (mV), not raw ADC counts
        ann_samples=np.asarray(ann.sample),
        ann_symbols=list(ann.symbol),
    )
