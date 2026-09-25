# ecg/

The project package.

Reading rule: in every folder, read the files that import nothing from the project first. The file that calls the others comes last. Each folder README lists its files in that order.

Start with `constants.py`. The four shared modules import nothing from the project:

1. `constants.py`: DS1/DS2 record lists, class names, symbol to class map, exclusions.
2. `config.py`: loads a YAML config and inlines its `data:` reference.
3. `seed.py`: seeding.
4. `logging_utils.py`: logger factory.

Then the subpackages, in pipeline order:

1. `data/`: Data Preparation (CRISP-DM 3)
2. `eda/`: Data Understanding (CRISP-DM 2)
3. `datasets/`: torch datasets and augmentation
4. `models/`: baseline and CNN (CRISP-DM 4)
5. `training/`: training loop and helpers
6. `evaluation/`: metrics and error analysis (CRISP-DM 5)
7. `noise/`: noise-robustness experiment
8. `deployment/`: MAX78002 toolchain helpers (CRISP-DM 6)

How they connect:

- `constants.py` is imported by `data/labels.py`, `data/splits.py`, `data/storage.py`, all of `eda/`, and `evaluation/metrics.py`. Nothing else defines record lists or class names.
- `config.py` and `seed.py` are called only by the scripts in `src/`, before any package function runs.
- `data/` reads `data/raw/` and writes `data/processed/`. `eda/` reads the same raw records through `data/records.py` and `data/labels.py`.
- `datasets/` reads `data/processed/`. `training/` uses `datasets/` and `models/`. `evaluation/` reads a checkpoint from `training/` and a split from `data/storage.py`.
- `noise/` reuses `data/` for preprocessing and `evaluation/` for metrics. `deployment/` reads `data/processed/` and the toolchain logs in `max78002/`.
