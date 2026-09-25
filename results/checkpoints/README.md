# results/checkpoints/

Trained model weights (`*.pth`, `*.pkl`), named `<config name>_best.*`. Git-ignored.

How they connect: `src/train.py` writes them through `src/ecg/training/checkpoint.py`, together with the config that produced them. `src/evaluate.py` and `src/noise_test.py` read them. The CNN checkpoint is also the input to `quantize.py` in the MAX78002 toolchain.
