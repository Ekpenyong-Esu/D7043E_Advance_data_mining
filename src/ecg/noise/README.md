# ecg/noise/

Loads the Noise Stress Test Database records. In Submission 1 they are only shown in a figure (report section 2.3).

Start with `nstdb.py`. It is the only file.

1. `nstdb.py`: loads the `bw`, `em` and `ma` noise records.

How it connects: `notebooks/EDA.ipynb` calls `nstdb.load_noise` to draw the noisy-ECG figure. The noise-robustness experiment that adds this noise to DS2 is part of Submission 2.
