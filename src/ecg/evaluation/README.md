# ecg/evaluation/

Metrics, predictions, error analysis and the figures of report sections 4 to 6.

Start with `metrics.py`. Read in this order:

1. `metrics.py`: accuracy, macro-F1, per-class precision, recall and F1. `flat_scores` gives them all in one dict.
2. `confusion.py`: the 4 x 4 confusion matrix as an array, a labelled table, or a CSV file.
3. `report.py`: `metrics_row` builds one row of `results/metrics.csv`; `upsert_row` writes it.
4. `errors.py`: picks misclassified beats per (true, predicted) pair for report section 5.5.
5. `predict.py`: turns a checkpoint into one function `predict(X, rr)`. `cnn_predictor` loads the FP32 or the INT8 checkpoint (INT8 runs in the ai8x integer simulation); `baseline_predictor` loads a joblib file. `cnn_scores` returns the raw outputs for the known-answer test of section 6.3.
6. `plots.py`: learning curve, confusion matrix, noise curve, error examples, FP32 versus INT8 bars.

How they connect: `notebooks/MODELING.ipynb` builds predictors with `predict.py`, scores DS2 with `metrics.py` and `report.py`, and draws every figure with `plots.py`. `ecg/noise/experiment.py` and `ecg/training/runner.py` use `metrics.py` too.
