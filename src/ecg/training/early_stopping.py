"""Stop when the selection metric has not improved for ``patience`` epochs."""


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience, self.mode = patience, mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience
