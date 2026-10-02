# configs/

YAML files with the settings each script reads.

Start with `preprocess.yaml`. It is the only config in Submission 1.

1. `preprocess.yaml`: validation records, lead, beat window, filters, normalisation, seed.

How it connects:

- `src/prepare_data.py` and `src/make_eda_figures.py` both load it with `src/ecg/config.py`.
- The DS1 and DS2 record lists are not here. They are fixed, so they are in `src/ecg/constants.py`.
