# ecg/models/

Baseline and CNN (report section 4).

Start with `baseline.py`. The three model files stand alone. `registry.py`, which picks one of them, comes last.

1. `baseline.py`: random forest or logistic regression on flattened beats plus RR features.
2. `cnn.py`: plain PyTorch CNN using Conv1d, pooling, ReLU and Linear only.
3. `ai8x_cnn.py`: the same layers with ai8x wrappers, for QAT and synthesis. Needs `ai8x-training` installed.
4. `registry.py`: turns `cfg['model']` into one of the models above.
