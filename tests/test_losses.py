"""Check the class weights and the focal loss."""
import numpy as np
import torch

from ecg.training.early_stopping import EarlyStopping
from ecg.training.losses import FocalLoss, build_loss, class_weights


def test_class_weights_favour_rare_classes():
    y = np.array([0] * 900 + [1] * 50 + [2] * 40 + [3] * 10)
    w = class_weights(y, 4, 0.5)
    assert w[3] > w[1] > w[0] and torch.isclose(w.mean(), torch.tensor(1.0))


def test_focal_loss_is_smaller_for_confident_correct_predictions():
    logits = torch.tensor([[6.0, 0.0, 0.0, 0.0]]); target = torch.tensor([0])
    ce = torch.nn.functional.cross_entropy(logits, target)
    assert FocalLoss(2.0)(logits, target) < ce


def test_build_loss_kinds():
    y = np.array([0, 0, 0, 1, 2, 3])
    assert isinstance(build_loss({"loss": "ce"}, y, 4), torch.nn.CrossEntropyLoss)
    assert build_loss({"loss": "weighted_ce", "class_weight_power": 0.5}, y, 4).weight is not None
    assert isinstance(build_loss({"loss": "focal"}, y, 4), FocalLoss)


def test_early_stopping():
    es = EarlyStopping(patience=2)
    assert [es.step(v) for v in (0.5, 0.6, 0.55, 0.5)] == [False, False, False, True]
    assert es.best == 0.6
