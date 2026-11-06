# data/processed/

One compressed NumPy file per split, written by section 0.2 of `notebooks/EDA.ipynb` through `src/ecg/data/pipeline.py` and `storage.py`. `../processed_w512/` has the same layout with L = 512.

Start with `train.npz`. All three files have the same keys:

| key | shape | meaning |
|---|---|---|
| `X` | (n, 1, L) float32 | filtered, normalised beat windows, clipped to [-8, 8] |
| `y` | (n,) int64 | class index: 0=N, 1=S, 2=V, 3=F |
| `record_ids` | (n,) | source record of each beat |
| `rr` | (n, 3) float32 | pre-RR, post-RR (seconds, capped at 3 s), pre-RR divided by local mean RR |

How they connect: the preprocessing step of the notebook writes all three. `src/ecg/datasets/loaders.py` reads them for training (`X` divided by 8 is the model input; `rr` is used by the baselines only). `test.npz` is DS2: `load_split` refuses it unless `final=True` is passed, and only section 5 of `notebooks/MODELING.ipynb` does that.
