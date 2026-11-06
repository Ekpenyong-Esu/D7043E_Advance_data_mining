"""Check the network YAML, the sample export and the evidence parser. No toolchain needed."""
import numpy as np
import yaml

from ecg.deployment.evidence import data_memory_bytes, evidence_table, expected_output, memory_usage
from ecg.deployment.network_yaml import network_layers, processor_mask, write_network_yaml
from ecg.deployment.sample_input import export_sample

LOG = """4x1x1 FULL-RES OUTPUT:
[  45216  -39588   35379 -255252]
RESOURCE USAGE
Weight memory: 25,168 bytes out of 2,396,160 bytes total (1.1%)
Bias memory:   180 bytes out of 8,192 bytes total (2.2%)
"""


def test_network_yaml(model_cfg, tmp_path):
    assert processor_mask(1) == "0x0000000000000001" and processor_mask(64) == "0xffffffffffffffff"
    layers = network_layers(model_cfg)
    assert [L["operation"] for L in layers] == ["Conv1d"] * 4 + ["MLP"]
    assert layers[1]["max_pool"] == 2 and layers[3]["max_pool"] == 4 and layers[4]["flatten"] is True
    assert layers[0]["out_offset"] != "0x0000"                 # the input sits at 0, so layer 0 must write elsewhere
    assert all(a["out_offset"] != b["out_offset"] for a, b in zip(layers, layers[1:]))
    text = write_network_yaml({"arch": "ai87ecgnet", "dataset": "ecg_mitbih", "model": model_cfg}, tmp_path / "n.yaml")
    assert yaml.safe_load(text)["arch"] == "ai87ecgnet" and "processors: 0xffffffffffffffff" in text


def test_sample_export(tmp_path):
    beat = np.linspace(-8, 8, 256, dtype=np.float32)[None]
    x = export_sample(beat, 8.0, tmp_path / "s.npy")
    assert x.shape == (1, 256) and x.dtype == np.int64 and x.min() == -128 and x.max() == 127
    assert np.array_equal(np.load(tmp_path / "s.npy"), x)


def test_evidence_parsing(model_cfg, tmp_path):
    assert memory_usage(LOG)["weight_bytes"] == 25168
    assert expected_output(LOG).tolist() == [45216, -39588, 35379, -255252]
    assert data_memory_bytes(model_cfg, 256) == 16 * 256
    t = evidence_table(True, True, tmp_path, LOG, 4096, np.array([45216, -39588, 35379, -255252]))
    assert t.loc[t["Requirement"].str.startswith("Known-answer"), "Result"].iloc[0] == "PASS"
    assert t.loc[t["Requirement"] == "C/device files generated", "Result"].iloc[0] == "FAIL"  # no files in tmp_path
