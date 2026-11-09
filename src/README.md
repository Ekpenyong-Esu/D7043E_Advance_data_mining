# src/

The `ecg/` package. There are no scripts here: the two notebooks in `notebooks/` are the only entry points and call the package directly. The script names in the assignment (`prepare_data.py`, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `noise_test.py`) map to the subpackages `ecg/data/`, `ecg/datasets/`, `ecg/models/`, `ecg/training/`, `ecg/evaluation/` and `ecg/noise/`.

Start with `ecg/README.md`, then the subpackage READMEs in the order it lists.

How it connects:

- `notebooks/EDA.ipynb` loads `configs/preprocess.yaml` with `ecg/config.py`, sets the seed with `ecg/seed.py`, then calls `ecg/data/pipeline.py`. That writes `data/processed/` and the tables in `results/`. It then calls `ecg/eda/` for the figures and tables of report section 2.
- `notebooks/MODELING.ipynb` loads `configs/model.yaml` with `ecg/config.py`, then calls `ecg/training/` (baselines, sweep, CNN, QAT), `ecg/deployment/` (quantisation, synthesis), `ecg/evaluation/` (DS2 scores, errors, figures) and `ecg/noise/` (noise test). It writes `results/` and `max78002/`.
- The report sections quote the tables in `results/` and `max78002/`.

Design rules:

1. The notebooks only call the package. They have no logic of their own.
2. `ecg/constants.py` is the only place where record lists and the label map are written.
3. Every number you can tune is in `configs/`. No module hard-codes a window size, a cut-off or a channel count.
4. Functions take arrays and return arrays. File IO is only in `records.py`, `storage.py`, `pipeline.py`, `loaders.py`, `checkpoint.py`, the two log writers and `deployment/`.
5. `data/pipeline.py` and `training/runner.py` only call the other modules. They have no algorithm of their own.
6. Keep each file under about 80 lines. Split it if it grows past that.
