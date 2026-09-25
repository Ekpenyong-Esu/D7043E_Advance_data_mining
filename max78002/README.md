# max78002/

Output from the MAX78002 toolchain, used as evidence in report section 6. No board is used.

Start with `model_configuration/`. The three folders follow the toolchain in order:

1. `model_configuration/`: the ai8x network YAML and the dataset and model files added to `ai8x-training`.
2. `quantization_output/`: the `quantize.py` log and the quantised checkpoint (`*-q.pth.tar`).
3. `synthesis_output/`: the `ai8xize.py` output (generated C code, `weights.h`) and its log.

How they connect:

- The model and dataset files in `model_configuration/` are copied into `ai8x-training`, which trains the network and writes a checkpoint.
- `quantize.py` from `ai8x-synthesis` turns that checkpoint into the INT8 checkpoint stored in `quantization_output/`.
- `ai8xize.py` takes the INT8 checkpoint, the network YAML from `model_configuration/` and a sample beat from `src/ecg/deployment/sample_input.py`. It writes the C code and log in `synthesis_output/`.
- `src/ecg/deployment/evidence.py` reads the logs in the last two folders to fill the PASS/FAIL table in report section 6.3.

The commands are in `STEP_BY_STEP.md`, Step 9.
