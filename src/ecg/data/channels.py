"""Pick one ECG lead from a record by its name.

Most records have MLII as the first channel, but record 114 has it as the
second channel. Picking by name instead of by position avoids that trap.
"""
import numpy as np

from ecg.data.records import Record


class LeadNotAvailable(ValueError):
    """Raised when a record does not have the lead we asked for."""


def select_lead(record: Record, lead: str) -> np.ndarray:
    """Return the 1-D signal of ``lead``. Records 102 and 104 have no MLII and raise."""
    if lead not in record.sig_names:
        raise LeadNotAvailable(f"Record {record.record_id} has leads {record.sig_names}, not {lead}")
    return record.signal[:, record.sig_names.index(lead)]
