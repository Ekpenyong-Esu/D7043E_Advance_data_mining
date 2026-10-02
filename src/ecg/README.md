# ecg/

The project package.

Reading rule: in every folder, read the files that import nothing from the project first. The file that calls the others comes last. Each folder README lists its files in that order.

Start with `constants.py`. The four shared modules import nothing from the project:

1. `constants.py`: DS1/DS2 record lists, class names, symbol to class map, exclusions.
2. `config.py`: loads a YAML config.
3. `seed.py`: seeding.
4. `logging_utils.py`: logger factory.

Then the subpackages:

1. `data/`: Data Preparation (CRISP-DM 3)
2. `eda/`: Data Understanding (CRISP-DM 2)
3. `noise/`: loads the noise records for the noisy-ECG figure

How they connect:

- `constants.py` is imported by `data/labels.py`, `data/splits.py`, `data/storage.py`, `data/pipeline.py` and all of `eda/`. Nothing else defines record lists or class names.
- `config.py` and `seed.py` are called only by the scripts in `src/`, before any package function runs.
- `data/` reads `data/raw/` and writes `data/processed/`. `eda/` reads the same raw records through `data/records.py`, `data/channels.py` and `data/labels.py`.
- `noise/nstdb.py` reads `data/raw/nstdb/`. `src/make_eda_figures.py` and the notebook use it.
