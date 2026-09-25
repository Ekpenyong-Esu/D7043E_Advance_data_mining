# max78002/model_configuration/

The network YAML for `ai8xize.py`, plus copies of the dataset loader and model file registered in `ai8x-training`.

Start with the network YAML. It must match the layers in `src/ecg/models/ai8x_cnn.py`. The dataset loader reads `data/processed/*.npz`. The outputs of training go to `../quantization_output/` next.
