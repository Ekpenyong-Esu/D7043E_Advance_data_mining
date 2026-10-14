# data/

Raw and processed data. The data files are git-ignored; only the README files are kept.

Start with `raw/README.md`, then `processed/README.md`.

1. `raw/`: PhysioNet downloads (`mitdb/`, `nstdb/`).
2. `processed/`: `train.npz`, `val.npz`, `test.npz`.

How they connect: `make data` downloads both databases with wfdb and fills `raw/`. `make prepare` runs `src/prepare_data.py`, which reads `raw/mitdb/` and writes `processed/`. `notebooks/EDA.ipynb` reads `raw/` for the figures and tables, and one beat from `processed/` for the augmentation figure.
