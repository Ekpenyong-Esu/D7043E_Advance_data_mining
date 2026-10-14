# src/

Entry-point scripts and the `ecg/` package.

Start with `ecg/README.md`, then read the scripts in the order you run them:

1. `prepare_data.py`: raw records to processed splits (`make prepare`).

The EDA figures and tables are made by `notebooks/EDA.ipynb`, which calls `ecg/eda/`.

How they connect:

- Each script loads the config with `ecg/config.py` and then calls the package. `prepare_data.py` also sets the seed with `ecg/seed.py`.
- `prepare_data.py` calls `ecg/data/pipeline.py`. It writes `data/processed/` and the tables in `results/`.
- The report sections quote the tables both scripts write to `results/`.

Design rules:

1. The scripts in `src/` only parse arguments and call the package.
2. `ecg/constants.py` is the only place where record lists and the label map are written.
3. Every number you can tune is in `configs/`. No module hard-codes a window size or a cut-off.
4. The functions in `ecg/data/` take arrays and return arrays. File IO is only in `records.py`, `storage.py` and `pipeline.py`.
5. `data/pipeline.py` only calls the other modules. It has no algorithm of its own.
6. Keep each file under about 80 lines. Split it if it grows past that.
