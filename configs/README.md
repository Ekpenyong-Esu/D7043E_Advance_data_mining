# configs/

YAML files with the settings each script reads.

Start with `preprocess.yaml`. Then read the others in this order:

1. `preprocess.yaml`: records per split, lead, beat window, filters, normalisation, seed.
2. `baseline.yaml`: classical baseline (random forest or logistic regression), report section 4.1.
3. `cnn_fp32.yaml`: 1D CNN, float training.
4. `cnn_qat.yaml`: the same CNN with quantisation-aware training.
5. `noise.yaml`: noise-robustness experiment (checkpoint, noise types, SNR levels).

How they connect:

- `preprocess.yaml` is read by `src/prepare_data.py` and `src/make_eda_figures.py`.
- The other four each have a `data:` line that points to `preprocess.yaml`. `src/ecg/config.py` loads a config and copies the preprocessing settings into it, so they are defined once.
- `baseline.yaml`, `cnn_fp32.yaml` and `cnn_qat.yaml` are passed to `src/train.py`. Their `model:` block is read by `src/ecg/models/registry.py`.
- `noise.yaml` is passed to `src/noise_test.py`. Its `checkpoint:` line names a file in `results/checkpoints/`.
