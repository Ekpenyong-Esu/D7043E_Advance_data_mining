# ECG Heartbeat Classification for Edge AI (MAX78002): Submission 1

Course project for D7043E at Luleå University of Technology (LTU). The full project builds a 1D CNN that classifies single ECG heartbeats from the MIT-BIH Arrhythmia Database into four classes (N, S, V, F), and prepares it for the Analog Devices MAX78002. The work follows CRISP-DM.

This package is Submission 1. It covers the first three CRISP-DM stages: business understanding, data understanding and data preparation. No model is trained yet.

This is a course project. Nothing here is a medical device and nothing is clinically validated. No physical MAX78002 board is used.

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

| Dataset | Use in Submission 1 | Source |
|---------|---------------------|--------|
| MIT-BIH Arrhythmia Database (`mitdb`) | Training, validation and test data | https://physionet.org/content/mitdb/1.0.0/ |
| MIT-BIH Noise Stress Test Database (`nstdb`) | Noisy-ECG examples in the EDA | https://physionet.org/content/nstdb/1.0.0/ |

mitdb has 48 records from 47 subjects, 2 channels, 360 Hz, about 30 minutes each, and about 110 000 expert beat annotations.

Raw data is not committed to this repository. See Section 5 for download instructions.

### Inter-patient split (de Chazal protocol)

Records are assigned whole to one split. Beats from one record never appear in two splits.

| Split | Records |
|-------|---------|
| DS1 (train and validation, split by record) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 |
| DS2 (final test, untouched until final evaluation) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |

The validation records within DS1 are listed in `configs/preprocess.yaml`. The rest of DS1 is the training set.

## 3. Repository layout

```
D7043E/
├── README.md              this file: install, download, reproduce
├── report.pdf             the report, CRISP-DM stages 1 to 3
├── requirements.txt
├── pyproject.toml         pytest, ruff and mypy settings
├── Makefile               data, prepare, eda, test, report
├── configs/               preprocess.yaml, the settings every script reads
├── data/                  raw and processed data, git-ignored
├── notebooks/             EDA.ipynb
├── src/                   entry-point scripts and the ecg/ package
├── tests/                 pytest
├── results/               tables written by the pipeline
└── report/                LaTeX sources, compiles to report.pdf
```

Each folder has a short README that says which file to read first, what every file is for, and how the files connect.

### Where to start reading

1. This file, for setup and the commands.
2. `configs/README.md`, then `configs/preprocess.yaml`, for the settings everything reads.
3. `src/README.md`, then `src/ecg/README.md`, then the subpackage READMEs in the order they list.
4. `results/README.md` and `report/README.md`, to see where the outputs go.

### How the pieces connect

- `configs/preprocess.yaml` holds the settings. Both scripts in `src/` start by loading it.
- `src/prepare_data.py` reads `data/raw/` and writes `data/processed/` and the tables in `results/`.
- `src/make_eda_figures.py` reads `data/raw/` and writes `results/eda/` and `report/figures/`.
- `notebooks/EDA.ipynb` shows the same analysis. It calls the same functions as `src/make_eda_figures.py`.
- `report/` includes figures from `report/figures/` and quotes the numbers in `results/`.

## 4. Installation

Python 3.10 or 3.11. `uv` is used because `python -m venv` may lack `ensurepip` on WSL.

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name d7043e --display-name "Python (D7043E)"
```

Main dependencies: `wfdb`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pyyaml`, `jupyter`.

## 5. Downloading the data

Download both databases into `data/raw/`. Do not commit them.

```bash
# Option A: the Makefile target (uses the wfdb helper)
make data

# Option B: the wfdb helper by hand
mkdir -p data/raw
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"

# Option C: wget
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/mitdb https://physionet.org/files/mitdb/1.0.0/
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/nstdb https://physionet.org/files/nstdb/1.0.0/
```

If the download fails with an SSL certificate error (common behind a proxy or in WSL), point Python at the system certificate store first. `make data` already does this.

```bash
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
```

Expected layout afterwards:

```
data/raw/mitdb/100.dat  100.hea  100.atr  ...
data/raw/nstdb/bw.dat   em.dat   ma.dat   ...
```

## 6. Reproducing the preprocessing

The random seed and all preprocessing parameters are in `configs/preprocess.yaml`.

```bash
# 1. Preprocess and build the splits. Writes data/processed/*.npz and the tables in results/
make prepare
# the same without make:
python src/prepare_data.py --config configs/preprocess.yaml

# 2. EDA figures and tables for report section 2
make eda
# or open the notebook
jupyter notebook notebooks/EDA.ipynb

# 3. Run the tests
make test

# 4. Build report.pdf
make report
```

Outputs:

| File | Content |
|------|---------|
| `data/processed/train.npz`, `val.npz`, `test.npz` | The beats, labels, record ids and RR features of each split |
| `results/split_stats.csv` | Beats per class per split (report table 3.8) |
| `results/record_stats.csv` | The same per record, plus dropped and excluded counts |
| `results/label_mapping.json` | The symbol to class mapping |
| `results/eda/` | Tables for report section 2 |
| `report/figures/` | Figures for report sections 2 and 3 |

### Preprocessing steps

1. Read the MLII lead of each record at 360 Hz.
2. Remove baseline wander with two median filters, then remove fast noise with a 40 Hz low-pass filter.
3. Cut a fixed window of 256 samples around each annotated R-peak.
4. Normalise each beat.
5. Clip extreme values.
6. Compute the RR-interval features of each beat.

The exact parameters (window size, filter cut-offs, normalisation method, seed) are in `configs/preprocess.yaml` and in report section 3.9.

## 7. Edge-AI constraints (MAX78002)

The model comes in Submission 2. It must meet these limits, and they already shape the data preparation:

- Model size under 2 MB, INT8 weights.
- Only accelerator-supported operations: `Conv1d`, `MaxPool1d` or `AvgPool1d`, `ReLU`, a small `Linear` output layer.
- Fixed-size input. This is why every beat is cut to the same 256 samples.
- A bounded input range. This is why every beat is normalised and clipped.

## 8. Group

| Name | Programme | Role |
|------|-----------|------|
|      |           |      |

## 9. References

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol, 2001.
- Moody GB, Muldrow WE, Mark RG. *A noise stress test for arrhythmia detectors.* Computers in Cardiology, 1984.
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE TBME, 2004. (DS1/DS2 protocol)
- Analog Devices, MAX78002 documentation: https://www.analog.com/en/products/max78002.html
- WFDB Python package: https://github.com/MIT-LCP/wfdb-python
