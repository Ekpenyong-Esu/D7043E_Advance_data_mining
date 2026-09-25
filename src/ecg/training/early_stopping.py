"""Stop training when the validation score has not improved for a while.

This avoids wasting time and avoids overfitting to the training data.
"""


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        """``patience`` is how many bad epochs to allow. ``mode`` is "max" if higher is better."""
        self.patience, self.mode = patience, mode
        self.best: float | None = None  # best score seen so far
        self.bad_epochs = 0             # epochs in a row without improvement

    def step(self, value: float) -> bool:
        """Record this epoch's score. Return True when training should stop."""
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
