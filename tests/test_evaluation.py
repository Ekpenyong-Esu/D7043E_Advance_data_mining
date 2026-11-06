"""Check the metrics, the confusion matrix and the error picker."""
import numpy as np

from ecg.evaluation.confusion import as_frame, compute
from ecg.evaluation.errors import error_counts, select_examples
from ecg.evaluation.metrics import flat_scores, per_class, summary


def test_summary_and_per_class():
    y = np.array([0, 0, 1, 2, 3]); p = np.array([0, 0, 0, 2, 3])
    s = summary(y, p)
    assert np.isclose(s["accuracy"], 0.8)
    pc = {r["class"]: r for r in per_class(y, p)}
    assert pc["S"]["recall"] == 0.0 and pc["V"]["f1"] == 1.0  # the S beat was missed
    assert np.isclose(flat_scores(y, p)["f1_V"], 1.0)


def test_confusion_matrix_and_errors():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 1, 0, 1])
    cm = compute(y, p)
    assert as_frame(cm).loc["N", "S"] == 1 and cm.sum() == 4
    picks = select_examples(y, p, [("N", "S"), ("S", "N"), ("V", "N")], k=3)
    assert picks[("N", "S")].tolist() == [1] and len(picks[("V", "N")]) == 0
    assert error_counts(y, p, [("N", "S"), ("S", "N")]) == {"N->S": 1, "S->N": 1}
