"""Entry point: turn the raw PhysioNet records into processed train/val/test arrays.

Run with ``make prepare`` or ``python src/prepare_data.py --config configs/preprocess.yaml``.
This file only reads the arguments. The real work is in ``ecg.data.pipeline``.
"""
import argparse

from ecg.config import load_config
from ecg.data.pipeline import prepare_all_splits
from ecg.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/preprocess.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])       # same seed, same output
    prepare_all_splits(cfg)


if __name__ == "__main__":
    main()
