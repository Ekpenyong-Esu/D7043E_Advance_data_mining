# ecg/deployment/

Software-level deployment to the MAX78002 with the Analog Devices toolchain (report section 6).

Start with `toolchain.py`. Read in this order:

1. `toolchain.py`: finds the two cloned repositories (`make toolchain`). `import_ai8x` imports the layer wrappers and selects the MAX78002, in FP32 or in INT8-simulation mode. `run_quantize` and `run_synthesis` run `quantize.py` and `ai8xize.py` in the synthesis venv and write their logs.
2. `network_yaml.py`: writes the layer description `ai8xize.py` needs, from the `model:` block of the config.
3. `sample_input.py`: exports one beat as the integer `.npy` sample that `ai8xize.py` turns into a known-answer test.
4. `evidence.py`: reads the synthesis log (memory usage, expected output) and builds the PASS/FAIL table of section 6.3.

How they connect: `notebooks/MODELING.ipynb` calls them in order. The FP32 or QAT checkpoint from `ecg/training/` goes into `run_quantize`, which writes the INT8 checkpoint into `max78002/quantization_output/`. That checkpoint, the YAML from `network_yaml.py` and the sample from `sample_input.py` go into `run_synthesis`, which writes the C project into `max78002/synthesis_output/`. `evidence.py` reads what came out. `ecg/evaluation/predict.py` uses `import_ai8x` to run the INT8 checkpoint in Python.
