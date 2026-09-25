# tests/

Pytest tests for the array-in, array-out functions. Run with `make test`. File IO and plotting are not tested.

Start with `conftest.py`. Then read the tests in pipeline order:

1. `conftest.py`: shared fixtures (seeded random generator).
2. `test_labels.py`: symbol to class mapping and exclusion counting.
3. `test_splits.py`: DS1/DS2 split is disjoint and complete; validation comes from DS1.
4. `test_filters.py`: baseline removal keeps the QRS, low-pass removes high frequencies.
5. `test_beats.py`: window length and boundary dropping.
6. `test_normalize.py`: z-score and clipping.
7. `test_rr.py`: a premature beat has a short pre-RR and a long post-RR.
8. `test_snr.py`: skipped until `ecg/noise/snr.py` is written.

How they connect: each test file checks one module with the same name in `src/ecg/data/` (or `src/ecg/noise/` for `test_snr.py`). The fixtures in `conftest.py` are shared by all of them. `pyproject.toml` tells pytest where `src/` is.
