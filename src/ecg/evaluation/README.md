# ecg/evaluation/

Metrics and error analysis (report section 5).

Start with `metrics.py`. The four helper files stand alone. `runner.py`, which calls all of them, comes last.

1. `metrics.py`: accuracy, macro-F1, per-class precision, recall and F1.
2. `confusion.py`: confusion matrix as CSV and figure.
3. `report.py`: appends a model's row to `results/metrics.csv`.
4. `errors.py`: picks and plots misclassified beats (section 5.5).
5. `runner.py`: loads a checkpoint and a split, predicts, writes the files above.
