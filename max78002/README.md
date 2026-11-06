# max78002/

Output of the Analog Devices toolchain, the evidence for report section 6. No board is used; everything here is software-verified only.

Start with `model_configuration/`. The three folders follow the toolchain in order:

1. `model_configuration/`: the network YAML that `ai8xize.py` reads, generated from `configs/model.yaml`, plus the model and dataset files that register the same network in `ai8x-training`.
2. `quantization_output/`: the `quantize.py` logs and the INT8 checkpoints (`*-q.pth.tar`), one from the FP32 weights (PTQ) and one after QAT.
3. `synthesis_output/`: the sample beat, the `ai8xize.py` log, and the generated C project (`cnn.c`, `cnn.h`, `weights.h`, `main.c`, `sampledata.h`, `sampleoutput.h`, `softmax.c`, `Makefile`).

How they connect:

- Section 4 of `notebooks/MODELING.ipynb` runs `quantize.py` on the checkpoints in `results/checkpoints/` and writes `quantization_output/`.
- Section 8 writes `model_configuration/ecg-max78002.yaml` with `src/ecg/deployment/network_yaml.py`, the sample beat with `sample_input.py`, and runs `ai8xize.py`, which writes `synthesis_output/ecg-max78002/`.
- `src/ecg/deployment/evidence.py` reads the synthesis log and fills `results/deployment_evidence.csv`.
- The two Python files in `model_configuration/` are not used by the notebook. They show how to train the same network with the official `train.py` of `ai8x-training`.
