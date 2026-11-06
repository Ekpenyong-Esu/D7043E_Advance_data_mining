# configs/

YAML files with the settings the notebooks read.

Start with `preprocess.yaml`. Read in this order:

1. `preprocess.yaml`: validation records, lead, beat window (256 samples), filters, normalisation, seed. Read by `notebooks/EDA.ipynb`.
2. `preprocess_w512.yaml`: the same with a 512-sample window and its own output folders. The sweep builds `data/processed_w512/` from it.
3. `model.yaml`: the CNN (channels, kernels, pools), training (optimizer, loss, balancing, early stopping), augmentation, QAT, output paths and toolchain paths. Read by `notebooks/MODELING.ipynb`. The `data:` key names the preprocessing config the model trains on.
4. `baselines.yaml`: the logistic-regression and random-forest settings.
5. `sweep.yaml`: the model-selection stages. Each run is a small set of overrides of `model.yaml`.
6. `noise.yaml`: noise types and SNR levels of the robustness experiment.

How they connect:

- `src/ecg/config.py` loads them. `load_model_config` reads `model.yaml`, applies the overrides of a sweep run, and attaches the preprocessing config named in `data:` (window length, clip limit, processed-data path).
- The DS1 and DS2 record lists are not here. They are fixed, so they are in `src/ecg/constants.py`.
- The final model is the best sweep run; its overrides are stored in `results/experiments.csv`, so the notebook can rebuild its config.
