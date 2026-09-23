from ecg.data.labels import map_symbols


def test_mapping_and_exclusions():
    labels, keep, excluded = map_symbols(["N", "A", "V", "F", "/", "+", "Q"])
    assert labels[keep].tolist() == [0, 1, 2, 3]
    assert excluded == {"/": 1, "Q": 1}
    assert not keep[5]  # '+' is a rhythm annotation, not a beat
