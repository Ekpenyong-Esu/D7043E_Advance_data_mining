# ecg/deployment/

MAX78002 toolchain helpers (report section 6).

Start with `sample_input.py`. Read in this order:

1. `sample_input.py`: exports one preprocessed beat as the int8-scaled `.npy` that `ai8xize.py --sample-input` needs.
2. `evidence.py`: parses the quantize and synthesis logs into the section 6.3 PASS/FAIL table.

How they connect: `sample_input.py` takes one beat from `data/processed/` and writes the `.npy` file that `ai8xize.py` reads. `evidence.py` reads the logs in `max78002/quantization_output/` and `max78002/synthesis_output/` after the toolchain has run. The ai8x network YAML is written by hand and lives in `max78002/model_configuration/`.

Neither file is written yet.
