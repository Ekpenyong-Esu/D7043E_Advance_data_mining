"""The noise-robustness experiment. Not written yet.

For each noise type and each SNR level: add noise to the raw DS2 signals, run the
same preprocessing as the pipeline, score the frozen model, and record the result.
Writes results/noise_results.csv and results/noise_results.png.
Training is never touched.
"""


def run_noise_experiment(cfg: dict) -> None:
    """Run the whole experiment as set in ``cfg`` (configs/noise.yaml)."""
    raise NotImplementedError
