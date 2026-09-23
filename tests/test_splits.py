import pytest

from ecg.constants import DS1, DS2
from ecg.data.splits import assert_disjoint, build_splits


def test_splits_are_disjoint_and_cover_ds1():
    s = build_splits([108, 116])
    assert_disjoint(s)
    assert sorted(s["train"] + s["val"]) == sorted(DS1)
    assert s["test"] == sorted(DS2)


def test_validation_must_be_in_ds1():
    with pytest.raises(ValueError):
        build_splits([100])
