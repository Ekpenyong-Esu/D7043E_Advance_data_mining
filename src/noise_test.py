"""Entry point: noise-robustness experiment (clean / 12 dB / 6 dB / 0 dB).

Thin CLI only. Logic lives in ``ecg.noise``.
"""
import argparse

from ecg.config import load_config
from ecg.noise.experiment import run_noise_experiment
from ecg.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/noise.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_noise_experiment(cfg)


if __name__ == "__main__":
    main()
