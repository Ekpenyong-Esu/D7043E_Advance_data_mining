"""Entry point: train a baseline or a CNN from a config file.

Run with ``python src/train.py --config configs/cnn_fp32.yaml``.
This file only reads the arguments. The real work is in ``ecg.training``.
"""
import argparse

from ecg.config import load_config
from ecg.seed import set_seed
from ecg.training.runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    run_training(cfg)


if __name__ == "__main__":
    main()
