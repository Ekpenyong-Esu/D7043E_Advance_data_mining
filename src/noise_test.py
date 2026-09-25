"""Entry point: the noise-robustness experiment (clean, 12 dB, 6 dB, 0 dB).

Run with ``python src/noise_test.py --config configs/noise.yaml``.
This file only reads the arguments. The real work is in ``ecg.noise``.
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
    set_seed(cfg["seed"])       # the noise offsets are random, so seed them
    run_noise_experiment(cfg)


if __name__ == "__main__":
    main()
