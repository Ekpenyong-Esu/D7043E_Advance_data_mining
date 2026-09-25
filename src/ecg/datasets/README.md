# ecg/datasets/

Torch data access (Submission 2).

Start with `beat_dataset.py`. Read in this order:

1. `beat_dataset.py`: wraps the processed arrays as a `Dataset`. Applies a transform if one is given.
2. `augment.py`: training-only transforms: amplitude scaling, time shift, baseline wander, Gaussian noise.
3. `samplers.py`: minority oversampling of S and F.
4. `loaders.py`: builds the train, val and test loaders.

How they connect: `loaders.build_loaders` reads `data/processed/*.npz` through `ecg/data/storage.py` and wraps each split in a `BeatDataset`. For the train loader only, it passes the transform from `augment.build_transform` and the sampler from `samplers.py`. `ecg/training/runner.py` calls `build_loaders`.

`augment.py`, `samplers.py` and `loaders.py` are not written yet.
