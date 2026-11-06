# ecg/datasets/

PyTorch data access for training (Submission 2).

Start with `beat_dataset.py`. Read in this order:

1. `beat_dataset.py`: `to_model_input` maps a stored beat (z-scored, clipped to [-8, 8]) to the chip range [-1, 1), or to integers in [-128, 127] for INT8 simulation. `BeatDataset` wraps the arrays and applies a transform if one is given.
2. `augment.py`: training-only transforms: amplitude scaling, time shift, baseline wander, Gaussian noise. `build_transform` reads the `augmentation:` block of `configs/model.yaml`.
3. `samplers.py`: minority oversampling. Each beat is drawn with weight (1 / class frequency) ** power.
4. `loaders.py`: `load_split` reads one `.npz` file and refuses the test split without `final=True`. `build_loaders` builds the train loader (transform and sampler) and the validation loader (neither).

How they connect: `ecg/training/runner.py` calls `build_loaders` for the CNN and `load_split` for the baselines. `ecg/evaluation/predict.py` and `ecg/deployment/sample_input.py` call `to_model_input`, so every model sees beats scaled in exactly one way.
