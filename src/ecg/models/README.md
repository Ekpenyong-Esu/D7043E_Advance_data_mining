# ecg/models/

Baselines and the 1D CNN (report sections 4.1 and 4.2).

Start with `cnn.py`. The three model files stand alone. `registry.py`, which picks one of them, comes last.

1. `cnn.py`: `ECGCNN`, the plain PyTorch network: fused (MaxPool1d, Conv1d, ReLU) blocks and one Linear layer. `block_lengths` computes the length after each block, `layer_table` writes the table of report section 4.2 from the config alone.
2. `ai8x_cnn.py`: `AI87ECGNet`, the same network built with the ai8x wrappers (`FusedMaxPoolConv1dReLU`, `FusedConv1dReLU`, `Linear`). This is the version that is trained, quantised and synthesised. It needs the toolchain (`make toolchain`).
3. `baseline.py`: logistic regression and random forest on the flattened beat plus the RR features, and `n_parameters` for the size column of the results table.
4. `registry.py`: `build_cnn(cfg, ai8x)` returns the ai8x network when the module is given, the plain one otherwise.

How they connect: `ecg/training/runner.py` calls `build_cnn` and `build_baseline`. `ecg/evaluation/predict.py` calls `build_cnn` again to load checkpoints. `ecg/deployment/network_yaml.py` and `evidence.py` use `block_lengths` to describe the same layers for the synthesiser. The two CNN files must keep the same layer names (`conv1` to `conv4`, `fc`) and shapes.
