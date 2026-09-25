"""Check the symbol-to-class mapping."""
from ecg.data.labels import map_symbols


def test_mapping_and_exclusions():
    labels, keep, excluded = map_symbols(["N", "A", "V", "F", "/", "+", "Q"])
    assert labels[keep].tolist() == [0, 1, 2, 3]   # N, A, V, F map to classes 0..3
    assert excluded == {"/": 1, "Q": 1}            # paced and unclassifiable beats are counted
    assert not keep[5]                             # '+' is a rhythm mark, not a beat
