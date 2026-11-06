# results/checkpoints/

Trained models, named after their run in `../experiments.csv`.

Start with the final CNN, `<final run>_fp32.pth.tar` (the run named first in `../experiments.csv` when sorted by validation macro-F1).

1. `<run>_fp32.pth.tar`: one FP32 CNN per sweep run, the best epoch of that run. Saved by `src/ecg/training/checkpoint.py` in the layout `quantize.py` reads (`state_dict`, `arch`, `epoch`, `extras`, plus the config).
2. `<final run>_qat.pth.tar`: the final CNN after quantisation-aware fine-tuning.
3. `baseline_lr.joblib`: the logistic-regression pipeline. `baseline_rf.joblib` (random forest) is git-ignored because of its size; section 1 of `notebooks/MODELING.ipynb` rebuilds it.

How they connect: `src/ecg/training/runner.py` writes them. `src/ecg/evaluation/predict.py` loads them for DS2, the noise test and the error analysis. `quantize.py` turns the FP32 and QAT checkpoints into the INT8 checkpoints in `max78002/quantization_output/`.
