# data/processed/

One compressed NumPy file per split, written by `src/prepare_data.py` through `src/ecg/data/storage.py`.

Start with `train.npz`. All three files have the same keys:

| key | shape | meaning |
|---|---|---|
| `X` | (n, 1, 256) float32 | filtered, normalised, clipped beat windows |
| `y` | (n,) int64 | class index: 0=N, 1=S, 2=V, 3=F |
| `record_ids` | (n,) | source record of each beat |
| `rr` | (n, 3) float32 | pre-RR, post-RR (seconds, capped at 3 s), pre-RR divided by local mean RR |

How they connect: `src/ecg/datasets/loaders.py` reads them for training. `src/ecg/evaluation/runner.py` reads them for evaluation. `src/ecg/deployment/sample_input.py` takes one beat from them for the toolchain. `make eda` reads one beat from `train.npz` for the augmentation figure.

`test.npz` is DS2. Do not use it for model selection. `src/evaluate.py` needs `--final` to read it.
