# ecg/noise/

The Noise Stress Test Database and the noise-robustness experiment (report section 5.4). Noise is added only at evaluation time and only to DS2.

Start with `nstdb.py`. Read in this order:

1. `nstdb.py`: loads the `bw`, `em` and `ma` noise records.
2. `snr.py`: scales a noise record to a target SNR in dB and adds it to a signal.
3. `experiment.py`: `noisy_split` builds the DS2 arrays from the raw records with noise added; `run_noise_experiment` scores every predictor on clean DS2 and on every (noise type, SNR) condition.

How they connect: `experiment.py` reads the raw DS2 records with `ecg/data/pipeline.read_signal`, adds noise with `snr.add_noise`, then runs `ecg/data/pipeline.beats_from_signal`, the same filtering, cutting and normalisation as in training. The predictors come from `ecg/evaluation/predict.py`. `notebooks/EDA.ipynb` uses `nstdb.py` for the noisy-ECG figure; `notebooks/MODELING.ipynb` runs the experiment.
