# data/

Raw and processed data. The data files are git-ignored; only the README files are kept.

Start with `raw/README.md`, then `processed/README.md`.

1. `raw/`: PhysioNet downloads (`mitdb/`, `nstdb/`).
2. `processed/`: `train.npz`, `val.npz`, `test.npz`.

How they connect: `make data` runs `src/ecg/data/download.py` and fills `raw/`. `make prepare` runs `src/prepare_data.py`, which reads `raw/mitdb/` and writes `processed/`. Training, evaluation and the noise test read `processed/`. The noise test also reads `raw/nstdb/`.
