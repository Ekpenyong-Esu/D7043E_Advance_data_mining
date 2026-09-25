# ecg/training/

Training loop and helpers (Submission 2).

Start with `loop.py`. The five helper files stand alone. `runner.py`, which calls all of them, comes last.

1. `loop.py`: one epoch of training or prediction, no file IO.
2. `losses.py`: class weights and focal loss.
3. `early_stopping.py`: stops when validation macro-F1 stops improving.
4. `checkpoint.py`: saves and loads weights with the config that produced them.
5. `experiment_log.py`: appends one row per run to `results/experiments.csv`.
6. `runner.py`: wires data, model, loss, loop and logging for one run.
