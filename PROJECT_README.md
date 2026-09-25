# ECG Heartbeat Classification for Edge AI (MAX78002)

Course project for D7043E at Luleå University of Technology (LTU). A 1D CNN classifies single ECG heartbeats from the MIT-BIH Arrhythmia Database into four classes (N, S, V, F). The work follows CRISP-DM and prepares the model for software-level deployment on the Analog Devices MAX78002.

This is a course project. The model is not a medical device and is not clinically validated. No physical MAX78002 board is used, so no hardware measurements (latency, power, energy, on-device accuracy) are claimed.

## 1. Project goal

Classify single ECG heartbeats into four AAMI classes while meeting the Edge-AI limits: model size under 2 MB, INT8 inference, and only operations the MAX78002 CNN accelerator supports.

| Class | Meaning                        |
|-------|--------------------------------|
| N     | Normal-type beats              |
| S     | Supraventricular ectopic beats |
| V     | Ventricular ectopic beats      |
| F     | Fusion beats                   |

Paced and unclassifiable beats (`/`, `f`, `Q`) are excluded. The exclusion is documented in the report and in `results/label_mapping.json`.

## 2. Datasets

| Dataset | Use | Source |
|---------|-----|--------|
| MIT-BIH Arrhythmia Database (`mitdb`) | Training, validation and test data | https://physionet.org/content/mitdb/1.0.0/ |
| MIT-BIH Noise Stress Test Database (`nstdb`) | Noise-robustness experiment | https://physionet.org/content/nstdb/1.0.0/ |

mitdb has 48 records from 47 subjects, 2 channels, 360 Hz, about 30 minutes each, and about 110 000 expert beat annotations.

Raw data is not committed to this repository. See Section 5 for download instructions.

### Inter-patient split (de Chazal protocol)

Records are assigned whole to one split. Beats from one record never appear in two splits.

| Split | Records |
|-------|---------|
| DS1 (train and validation, split by record) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 |
| DS2 (final test, untouched until final evaluation) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |

The train and validation records within DS1 are listed in `configs/preprocess.yaml`.

## 3. Repository layout

```
D7043E/
├── README.md              assignment text from the course page
├── PROJECT_README.md      this file: install, download, reproduce
├── STEP_BY_STEP.md        execution guide mapped to CRISP-DM
├── PROJECT_STRUCTURE.md   what every file is for
├── requirements.txt
├── pyproject.toml         pytest and ruff settings
├── Makefile               data, prepare, eda, baseline, train, qat, evaluate, noise, test, report
├── configs/               YAML settings for every script
├── data/                  raw and processed data, git-ignored
├── notebooks/             EDA.ipynb
├── src/                   entry-point scripts and the ecg/ package
├── tests/                 pytest
├── results/               tables, figures and checkpoints written by the pipeline
├── max78002/              toolchain output for report section 6
└── report/                LaTeX sources, compiles to report.pdf
```

Each folder has a short README that says which file to read first and how the files connect. `PROJECT_STRUCTURE.md` describes every file.

### Where to start reading

1. This file, for setup and the commands.
2. `STEP_BY_STEP.md`, for the order of work mapped to CRISP-DM.
3. `configs/README.md`, then `configs/preprocess.yaml`, for the settings everything reads.
4. `src/README.md`, then `src/ecg/README.md`, then the subpackage READMEs in the order they list.
5. `results/README.md` and `report/README.md`, to see where the outputs go.

### How the pieces connect

- `configs/` holds the settings. Every script in `src/` starts by loading one config.
- `src/prepare_data.py` reads `data/raw/` and writes `data/processed/` and the tables in `results/`.
- `src/make_eda_figures.py` reads `data/raw/` and writes `results/eda/` and `report/figures/`.
- `src/train.py` reads `data/processed/` and writes `results/checkpoints/`. `src/evaluate.py` and `src/noise_test.py` read a checkpoint and write `results/`.
- The MAX78002 toolchain takes the trained model and writes its output to `max78002/`.
- `report/` includes figures from `report/figures/`, and `src/fill_report_numbers.py` copies numbers from `results/` into the report text.

## 4. Installation

Python 3.10 or 3.11. `uv` is used because `python -m venv` may lack `ensurepip` on WSL.

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name d7043e --display-name "Python (D7043E)"
```

Main dependencies: `wfdb`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`, `pyyaml`, `jupyter`.

The MAX78002 toolchain (`ai8x-training`, `ai8x-synthesis`) is installed separately. See `STEP_BY_STEP.md`, Step 9.

## 5. Downloading the data

Download both databases into `data/raw/`. Do not commit them.

```bash
mkdir -p data/raw
# Option A: wfdb helper
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"

# Option B: wget
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/mitdb https://physionet.org/files/mitdb/1.0.0/
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/nstdb https://physionet.org/files/nstdb/1.0.0/
```

If the download fails with an SSL certificate error (common behind a proxy or in WSL), point Python at the system certificate store first:

```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

Expected layout afterwards:

```
data/raw/mitdb/100.dat  100.hea  100.atr  ...
data/raw/nstdb/bw.dat   em.dat   ma.dat   ...
```

## 6. Reproducing the pipeline

The random seed and all preprocessing parameters are in `configs/preprocess.yaml`.

```bash
# 1. Preprocess and build splits. Writes data/processed/*.npz and results/split_stats.csv
python src/prepare_data.py --config configs/preprocess.yaml

# 2. EDA figures and tables for report section 2
make eda
# or open the notebook
jupyter notebook notebooks/EDA.ipynb

# 3. Baseline
python src/train.py --config configs/baseline.yaml

# 4. 1D CNN, FP32
python src/train.py --config configs/cnn_fp32.yaml

# 5. Quantisation-aware training (INT8)
python src/train.py --config configs/cnn_qat.yaml

# 6. Evaluate on DS2, only after model selection is frozen
python src/evaluate.py --checkpoint results/checkpoints/cnn_qat_best.pth --split test

# 7. Noise robustness (clean, 12 dB, 6 dB, 0 dB)
python src/noise_test.py --checkpoint results/checkpoints/cnn_qat_best.pth
```

Outputs are written to `results/`.

### Preprocessing steps

1. Read the MLII lead of each record at 360 Hz.
2. Remove baseline wander with a filter.
3. Cut a fixed window around each annotated R-peak.
4. Normalise each beat.
5. Clip extreme values.

The exact parameters (window size, filter cut-offs, normalisation method, seed) are in `configs/preprocess.yaml` and in report section 3.9.

## 7. Edge-AI constraints (MAX78002)

- Model size under 2 MB, INT8 weights.
- Only accelerator-supported operations: `Conv1d`, `MaxPool1d` or `AvgPool1d`, `ReLU`, a small `Linear` output layer, fixed-size input.
- The network is defined with `ai8x` layer wrappers so it can be quantised with `quantize.py` and synthesised with `ai8xize.py`.
- No Transformers, LSTMs, GRUs, large ResNets or unsupported custom operations.

Toolchain output (acceptance, quantisation, synthesis, generated C files, weight and data memory) is stored under `max78002/` and reported in report section 6.

## 8. Group

| Name | Programme | Role |
|------|-----------|------|
|      |           |      |

## 9. References

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol, 2001.
- Moody GB, Muldrow WE, Mark RG. *A noise stress test for arrhythmia detectors.* Computers in Cardiology, 1984.
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE TBME, 2004. (DS1/DS2 protocol)
- Analog Devices, ai8x-training: https://github.com/analogdevicesinc/ai8x-training
- Analog Devices, ai8x-synthesis: https://github.com/analogdevicesinc/ai8x-synthesis
- Analog Devices, MAX78002 documentation: https://www.analog.com/en/products/max78002.html
- WFDB Python package: https://github.com/MIT-LCP/wfdb-python
