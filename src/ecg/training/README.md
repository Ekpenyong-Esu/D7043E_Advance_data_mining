# ecg/training/

Training loop and helpers (report section 4.3).

Start with `loop.py`. The five helper files stand alone. `runner.py`, which calls all of them, comes last.

1. `loop.py`: one epoch of training, prediction over a loader or an array, and the device choice. No file IO.
2. `losses.py`: class weights (inverse frequency to a power), focal loss, and `build_loss` for the `loss:` setting.
3. `early_stopping.py`: stops when validation macro-F1 has not improved for `early_stopping_patience` epochs.
4. `checkpoint.py`: saves and loads weights together with the config. The file layout is what `quantize.py` expects.
5. `experiment_log.py`: one row per run in `results/experiments.csv`; a rerun with the same name replaces its row.
6. `runner.py`: `run_cnn_training` (FP32, or QAT fine-tuning when `qat_from` is given) and `run_baseline`.

How they connect: `runner.py` builds the loaders (`ecg/datasets/`), the model (`ecg/models/`) and the loss, runs `fit` (the epoch loop), scores validation with `ecg/evaluation/metrics.py`, saves the best epoch with `checkpoint.py` and logs the run with `experiment_log.py`. `notebooks/MODELING.ipynb` is the only caller.
