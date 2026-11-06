"""Write the network description that ``ai8xize.py`` needs, from the ``model:`` block of the config.

The YAML lists every accelerator layer in order. For each layer it names the operation,
kernel size, padding, pooling, activation, which of the 64 processors read its input
(one processor per input channel) and where in data memory the output goes. The input is
loaded at offset 0, so the output offset alternates between 0x8000 and 0x0000: a layer
never overwrites its own input.
"""
from pathlib import Path

import yaml

OFFSETS = ("0x0000", "0x8000")


def processor_mask(n_channels: int) -> str:
    """Bit mask of the first ``n_channels`` processors, as the 16-digit hex string the tool expects."""
    return f"0x{(1 << n_channels) - 1:016x}"


def network_layers(model_cfg: dict) -> list[dict]:
    """One dict per layer, in the key order used by the Analog Devices examples."""
    m = model_cfg
    layers, in_ch = [], 1
    for i, (out_ch, k, pool, pad) in enumerate(zip(m["channels"], m["kernels"], m["pools"], m["paddings"])):
        layer: dict = {}
        if pool > 1:
            layer.update({"max_pool": pool, "pool_stride": pool})
        layer.update({"pad": pad, "activate": "ReLU", "out_offset": OFFSETS[(i + 1) % 2],
                      "processors": processor_mask(in_ch), "operation": "Conv1d", "kernel_size": k})
        if i == 0:
            layer["data_format"] = "CHW"  # single-channel input, one sample per byte
        layers.append(layer)
        in_ch = out_ch
    layers.append({"flatten": True, "out_offset": OFFSETS[(len(layers) + 1) % 2], "processors": processor_mask(in_ch),
                   "operation": "MLP", "output_width": 32, "activate": "None"})
    return layers


def write_network_yaml(cfg: dict, path: str | Path) -> str:
    """Write the YAML for ``cfg`` to ``path`` and return its text."""
    doc = {"arch": cfg["arch"], "dataset": cfg["dataset"], "layers": network_layers(cfg["model"])}
    text = ("---\n# MAX78002 network description for ai8xize.py, generated from configs/model.yaml by\n"
            "# src/ecg/deployment/network_yaml.py. One entry per accelerator layer, in order.\n"
            + yaml.safe_dump(doc, sort_keys=False, width=100))
    # Keep the processor masks unquoted hex, as in the Analog Devices examples.
    text = text.replace("processors: '", "processors: ").replace("'\n", "\n").replace("out_offset: '", "out_offset: ")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text)
    return text
