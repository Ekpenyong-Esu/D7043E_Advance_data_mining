"""Entry point: raw PhysioNet records -> processed train/val/test arrays.

Thin CLI only. All logic lives in ``ecg.data``.
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
    set_seed(cfg["seed"])
    prepare_all_splits(cfg)


if __name__ == "__main__":
    main()
