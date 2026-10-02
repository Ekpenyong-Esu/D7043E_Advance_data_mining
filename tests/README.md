# tests/

Pytest tests for the array-in, array-out functions. Run with `make test`. File IO and plotting are not tested.

Start with `test_labels.py`. Read the tests in pipeline order:

1. `test_labels.py`: symbol to class mapping and exclusion counting.
2. `test_splits.py`: DS1/DS2 split is disjoint and complete; validation comes from DS1.
3. `test_filters.py`: baseline removal keeps the QRS, low-pass removes high frequencies.
4. `test_beats.py`: window length and boundary dropping.
5. `test_normalize.py`: z-score and clipping.
6. `test_rr.py`: a premature beat has a short pre-RR and a long post-RR.

How they connect: each test file checks one module with the same name in `src/ecg/data/`. `pyproject.toml` tells pytest where `src/` is.
