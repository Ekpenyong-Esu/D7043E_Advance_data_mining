"""Assign whole records to train, validation and test. Beats are never split."""
from ecg.constants import DS1, DS2


def build_splits(validation_records: list[int]) -> dict[str, list[int]]:
    """Return the record lists for each split.

    Validation records must be taken from DS1. Training is the rest of DS1.
    Test is always the whole of DS2.
    """
    val = sorted(validation_records)
    unknown = set(val) - set(DS1)
    if unknown:
        raise ValueError(f"Validation records must come from DS1; got {sorted(unknown)}")
    train = sorted(set(DS1) - set(val))
    return {"train": train, "val": val, "test": sorted(DS2)}


def assert_disjoint(splits: dict[str, list[int]]) -> None:
    """Raise if any record is in more than one split. This guards against data leakage."""
    seen: set[int] = set()
    for name, recs in splits.items():
        overlap = seen & set(recs)
        if overlap:
            raise ValueError(f"Records {sorted(overlap)} appear in more than one split ({name})")
        seen |= set(recs)
