# data/

Raw and processed data. The data files are git-ignored; only the README files are kept.

Start with `raw/README.md`, then `processed/README.md`.

1. `raw/`: PhysioNet downloads (`mitdb/`, `nstdb/`).
2. `processed/`: `train.npz`, `val.npz`, `test.npz` with 256-sample beats (`configs/preprocess.yaml`).
3. `processed_w512/`: the same with 512-sample beats (`configs/preprocess_w512.yaml`), the window of the final model.

How they connect: section 0.1 of `notebooks/EDA.ipynb` downloads both databases with wfdb into `raw/`. Section 0.2 runs `src/ecg/data/pipeline.py`, which reads `raw/mitdb/` and writes `processed/`. The sweep in `notebooks/MODELING.ipynb` runs the same pipeline with the wide-window config to write `processed_w512/`. The EDA sections read `raw/` for the figures and tables. The noise experiment reads `raw/` again and adds `raw/nstdb/` noise before preprocessing.
