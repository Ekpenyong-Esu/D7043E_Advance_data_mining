# ecg/

The project package.

Reading rule: in every folder, read the files that import nothing from the project first. The file that calls the others comes last. Each folder README lists its files in that order.

Start with `constants.py`. The four shared modules import nothing from the project:

1. `constants.py`: DS1/DS2 record lists, class names, symbol to class map, exclusions.
2. `config.py`: loads a YAML config. `load_model_config` also attaches the preprocessing config a model config names.
3. `seed.py`: seeding (Python, NumPy, PyTorch).
4. `logging_utils.py`: logger factory.

Then the subpackages, in CRISP-DM order:

1. `data/`: Data Preparation (CRISP-DM 3), Submission 1.
2. `eda/`: Data Understanding (CRISP-DM 2), Submission 1.
3. `datasets/`: PyTorch data access, augmentation and oversampling, Submission 2.
4. `models/`: baselines and the 1D CNN (CRISP-DM 4).
5. `training/`: training loop, losses, checkpoints (CRISP-DM 4).
6. `evaluation/`: metrics, predictions, error analysis, figures (CRISP-DM 5).
7. `noise/`: the noise-robustness experiment (CRISP-DM 5).
8. `deployment/`: the MAX78002 toolchain (CRISP-DM 6).

How they connect:

- `constants.py` is imported by `data/`, `eda/`, `evaluation/` and `noise/`. Nothing else defines record lists or class names.
- `data/` reads `data/raw/` and writes `data/processed/`. `datasets/` reads `data/processed/`.
- `training/` uses `datasets/`, `models/` and `evaluation/metrics.py`, and writes `results/checkpoints/` and `results/experiments.csv`.
- `evaluation/predict.py` loads checkpoints through `models/` and `deployment/toolchain.py`. `noise/experiment.py` reuses `data/pipeline.py` on noisy signals and scores with `evaluation/`.
- `deployment/` turns the trained network into the INT8 checkpoint and the C project under `max78002/`.
- `notebooks/EDA.ipynb` calls `data/`, `eda/` and `noise/nstdb.py`. `notebooks/MODELING.ipynb` calls everything else.
