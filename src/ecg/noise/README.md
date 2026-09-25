# ecg/noise/

Noise-robustness experiment (report section 5.4). Noise is added only at evaluation time and only to DS2.

Start with `nstdb.py`. Read in this order:

1. `nstdb.py`: loads the `bw`, `em` and `ma` noise records.
2. `snr.py`: scales noise to a target SNR in dB and adds it to a signal.
3. `experiment.py`: runs the whole experiment and writes `results/noise_results.*`.

How they connect: `experiment.run_noise_experiment` reads the raw DS2 records with `ecg/data/records.py`, loads noise with `nstdb.load_noise`, mixes them with `snr.add_noise` at each SNR level, then runs the same preprocessing as `ecg/data/pipeline.py`. It loads the frozen model through `ecg/training/checkpoint.py` and scores it with `ecg/evaluation/metrics.py`. `nstdb.py` is also used by `src/make_eda_figures.py` for the noisy-ECG figure.

`snr.py` and `experiment.py` are not written yet. `tests/test_snr.py` is skipped until `snr.py` is done.
