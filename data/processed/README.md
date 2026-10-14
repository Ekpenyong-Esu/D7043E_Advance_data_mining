# data/processed/

One compressed NumPy file per split, written by `src/prepare_data.py` through `src/ecg/data/storage.py`.

Start with `train.npz`. All three files have the same keys:

| key | shape | meaning |
|---|---|---|
| `X` | (n, 1, 256) float32 | filtered, normalised, clipped beat windows |
| `y` | (n,) int64 | class index: 0=N, 1=S, 2=V, 3=F |
| `record_ids` | (n,) | source record of each beat |
| `rr` | (n, 3) float32 | pre-RR, post-RR (seconds, capped at 3 s), pre-RR divided by local mean RR |

How they connect: `make prepare` writes all three. `notebooks/EDA.ipynb` reads one beat from `train.npz` for the augmentation figure. The models in Submission 2 will train and test on these files.

`test.npz` is DS2. Do not use it for model selection.
