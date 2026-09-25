# src/

Entry-point scripts and the `ecg/` package.

Start with `ecg/README.md`, then read the scripts in the order you run them:

1. `prepare_data.py`: raw records to processed splits (`make prepare`).
2. `make_eda_figures.py`: figures and tables for report section 2 (`make eda`).
3. `fill_report_numbers.py`: writes numbers from `results/` into the report sections.
4. `train.py`: baseline or CNN training from a config.
5. `evaluate.py`: metrics on validation, or on test with `--final`.
6. `noise_test.py`: noise-robustness experiment.
7. `dataset.py`, `model.py`: re-exports kept for the layout the assignment asks for.

How they connect:

- Each script does the same three things: load a config with `ecg/config.py`, set the seed with `ecg/seed.py`, and call one function in the package.
- `prepare_data.py` calls `ecg/data/pipeline.py`. Its output in `data/processed/` and `results/` is the input for everything after it.
- `make_eda_figures.py` calls `ecg/eda/`. `fill_report_numbers.py` reads the tables both scripts wrote and edits `report/sections/`.
- `train.py` calls `ecg/training/runner.py`, which writes a checkpoint. `evaluate.py` and `noise_test.py` read that checkpoint through `ecg/evaluation/runner.py` and `ecg/noise/experiment.py`.
- `dataset.py` and `model.py` only import from `ecg/datasets/` and `ecg/models/`.
