# tests/

Pytest tests for the array-in, array-out functions. Run with `make test`. File IO, training and plotting are not tested, and nothing here needs the toolchain or the data.

Start with `test_labels.py`. Read the tests in pipeline order:

1. `test_labels.py`: symbol to class mapping and exclusion counting.
2. `test_splits.py`: DS1/DS2 split is disjoint and complete; validation comes from DS1.
3. `test_filters.py`: baseline removal keeps the QRS, low-pass removes high frequencies.
4. `test_beats.py`: window length and boundary dropping.
5. `test_normalize.py`: z-score and clipping.
6. `test_rr.py`: a premature beat has a short pre-RR and a long post-RR.
7. `test_augment.py`: the augmentations keep the beat shape and size.
8. `test_datasets.py`: input scaling to the chip range, dataset items, sampler weights, the DS2 lock.
9. `test_models.py`: CNN output shape, parameter count, only accelerator operations.
10. `test_losses.py`: class weights, focal loss, early stopping.
11. `test_evaluation.py`: metrics, confusion matrix, error picking.
12. `test_snr.py`: noise is added at the requested SNR.
13. `test_deployment.py`: network YAML, sample export, parsing of the synthesis log.

How they connect: each test file checks one module of `src/ecg/`. `conftest.py` holds the shared fixtures (a seeded random generator and the model config). `pyproject.toml` tells pytest where `src/` is. Both notebooks run the suite before building the report.
