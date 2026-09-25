# max78002/synthesis_output/

The folder `ai8xize.py` generates (`cnn.c`, `weights.h`, `main.c`, ...) and its log.

Start with the log. The inputs are the quantised checkpoint from `../quantization_output/`, the network YAML from `../model_configuration/`, and a sample beat from `src/ecg/deployment/sample_input.py`. The weight and data memory numbers for report section 6.3 come from the verbose log.
