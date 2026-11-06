"""Read the toolchain outputs and fill the PASS/FAIL table of report section 6.3.

The synthesiser writes ``log.txt`` next to the generated C files. Its last lines hold the
resource summary (weight and bias memory) and, above them, the output it expects from the
sample beat. ``sampleoutput.h`` carries the same expected output as C code, which the
generated ``main.c`` uses as a known-answer test on the device.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

C_FILES = ("cnn.c", "cnn.h", "weights.h", "main.c", "sampledata.h", "sampleoutput.h", "softmax.c", "Makefile")


def _n(value: int) -> str:
    """Thousands separated by a LaTeX thin space, as in the rest of the report."""
    return f"{int(value):,}".replace(",", "\\,")


def memory_usage(log_text: str) -> dict[str, int]:
    """Bytes used and available for weights and biases, from the ``RESOURCE USAGE`` lines."""
    out = {}
    for kind in ("Weight", "Bias"):
        m = re.search(rf"{kind} memory:\s+([\d,]+) bytes out of ([\d,]+) bytes", log_text)
        if m:
            out[f"{kind.lower()}_bytes"] = int(m.group(1).replace(",", ""))
            out[f"{kind.lower()}_capacity"] = int(m.group(2).replace(",", ""))
    return out


def expected_output(log_text: str) -> np.ndarray | None:
    """The synthesiser's own result for the sample beat: the last ``FULL-RES OUTPUT`` block of the log.

    The verbose log prints the output of every layer; the last block is the network output,
    one integer per class, before the software output scale.
    """
    blocks = re.findall(r"FULL-RES OUTPUT:\s*\[+([^\]]+)\]", log_text)
    return np.array([int(v) for v in re.findall(r"-?\d+", blocks[-1])]) if blocks else None


def output_scale(checkpoint: str | Path) -> int:
    """The power-of-two output scale the QAT bookkeeping left on the last layer (``fc.final_scale``).

    The accelerator produces the unscaled value; the device applies the scale in software, and the
    Python simulation applies it inside the model. Dividing the Python output by it gives the device value.
    """
    import torch  # local import keeps this module light for the tests

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    return int(2 ** int(state.get("fc.final_scale", torch.tensor(0.0)).item()))


def data_memory_bytes(model_cfg: dict, input_length: int) -> int:
    """Largest activation the chip must hold, in bytes (one byte per value): the biggest layer input or output."""
    from ecg.models.cnn import block_lengths  # local import keeps this module free of torch

    lengths = block_lengths(input_length, model_cfg["kernels"], model_cfg["pools"], model_cfg["paddings"])
    sizes = [input_length] + [c * L for c, L in zip(model_cfg["channels"], lengths)]
    return int(max(sizes))


def evidence_table(quant_ok: bool, synth_ok: bool, synth_dir: str | Path, log_text: str, data_bytes: int,
                   python_output: np.ndarray | None = None, python_scale: int = 1) -> pd.DataFrame:
    """The six rows of the deployment-evidence table, plus the known-answer comparison.

    ``python_output`` is the Python INT8 simulation of the sample beat and ``python_scale`` its
    software output scale (``output_scale``), so that it is compared at the accelerator's scale.
    """
    synth_dir = Path(synth_dir)
    files = [f for f in C_FILES if (synth_dir / f).exists()]
    mem = memory_usage(log_text)
    exp = expected_output(log_text)
    kat = "n/a"
    if exp is not None and python_output is not None:
        ours = np.round(np.asarray(python_output, dtype=np.float64) / python_scale).astype(np.int64)
        kat = "PASS" if np.array_equal(exp, ours) else "FAIL"
    rows = [
        ("Model accepted by MAX78002 toolchain", "PASS" if quant_ok and synth_ok else "FAIL",
         "ai8x layer checks, quantize.py and ai8xize.py read the model without error"),
        ("Quantization completed", "PASS" if quant_ok else "FAIL", "quantize.py log and the INT8 checkpoint"),
        ("Synthesis completed", "PASS" if synth_ok else "FAIL", "ai8xize.py exit code and log.txt"),
        ("C/device files generated", "PASS" if synth_ok and len(files) >= 5 else "FAIL", ", ".join(files)),
        ("Weight memory requirement", f"{_n(mem.get('weight_bytes', 0))} bytes of {_n(mem.get('weight_capacity', 0))}",
         "RESOURCE USAGE in log.txt"),
        ("Data memory requirement", f"{_n(data_bytes)} bytes (largest activation) of {_n(1_310_720)}",
         "largest layer input or output; the MAX78002 has 1\\,280 KiB of data memory"),
        ("Known-answer test: Python INT8 output equals synthesiser output", kat,
         "sampleoutput.h versus the Python INT8 simulation of the same beat"),
    ]
    return pd.DataFrame(rows, columns=["Requirement", "Result", "Evidence"])
