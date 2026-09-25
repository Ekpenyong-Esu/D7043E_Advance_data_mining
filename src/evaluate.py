"""Entry point: score a trained model on the validation or test split.

The test split is DS2. It must stay untouched until the very end, so
``--split test`` is refused unless ``--final`` is also given. This makes it
hard to peek at the test set by accident while choosing a model.
"""
import argparse

from ecg.config import load_config
from ecg.evaluation.runner import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None, help="Overrides the config checkpoint path")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--final", action="store_true", help="Required to evaluate on the test split")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        parser.error("Evaluating on DS2 requires --final. DS2 must stay untouched until final evaluation.")

    cfg = load_config(args.config)
    run_evaluation(cfg, split=args.split, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
