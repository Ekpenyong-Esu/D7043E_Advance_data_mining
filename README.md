# ECG Heartbeat Classification for Edge AI (MAX78002): Submission 2

Course project for D7043E at Luleå University of Technology (LTU). A 1D CNN classifies single ECG heartbeats from the MIT-BIH Arrhythmia Database into four classes (N, S, V, F) and is prepared for the Analog Devices MAX78002 with the official ai8x toolchain. The work follows CRISP-DM.

This package is Submission 2. It contains all six CRISP-DM stages: Submission 1 (business understanding, data understanding, data preparation) and the new modeling, evaluation and deployment work.

This is a course project. Nothing here is a medical device and nothing is clinically validated. No physical MAX78002 board is used, so no latency, power or on-device accuracy is measured. Everything in section 6 of the report is software-verified only.

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
| MIT-BIH Noise Stress Test Database (`nstdb`) | Noise-robustness experiment (added to DS2 at test time only) | https://physionet.org/content/nstdb/1.0.0/ |

mitdb has 48 records from 47 subjects, 2 channels, 360 Hz, about 30 minutes each, and about 110 000 expert beat annotations.

Raw data is not committed to this repository. See Section 5 for download instructions.

### Inter-patient split (de Chazal protocol)

Records are assigned whole to one split. Beats from one record never appear in two splits.

| Split | Records |
|-------|---------|
| DS1 (train and validation, split by record) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 |
| DS2 (final test, untouched until final evaluation) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |

The validation records within DS1 are listed in `configs/preprocess.yaml`: 124, 201, 205, 220, 223. The rest of DS1 is the training set. This is a Submission 2 revision of the Submission 1 choice (118, 124, 201, 203, 205); report section 3.1 explains why.

## 3. Repository layout

```text
D7043E/
├── README.md              this file: install, download, reproduce
├── BUILD_STEPS.md         what each notebook step does and why, in order
├── report.pdf             the report, CRISP-DM stages 1 to 6
├── requirements.txt
├── pyproject.toml         pytest, ruff and mypy settings
├── Makefile               run, eda, model, test, report, toolchain
├── configs/               preprocess.yaml, model.yaml, sweep.yaml, baselines.yaml, noise.yaml
├── data/                  raw and processed data, git-ignored
├── notebooks/             EDA.ipynb (Submission 1) and MODELING.ipynb (Submission 2), the two entry points
├── src/                   the ecg/ package
├── tests/                 pytest
├── results/               tables, figures, checkpoints and logs written by the notebooks
├── max78002/              toolchain output: network YAML, INT8 checkpoints, generated C code
├── toolchain/             the Analog Devices repositories, cloned by `make toolchain`, git-ignored
└── report/                LaTeX sources, compiles to report.pdf
```

Each folder has a short README that says which file to read first, what every file is for, and how the files connect.

### Where to start reading

1. This file, for setup and the commands.
2. `BUILD_STEPS.md`, for what each notebook step does.
3. `notebooks/README.md`, then `notebooks/EDA.ipynb` and `notebooks/MODELING.ipynb`. Their sections run in order.
4. `configs/README.md`, then the YAML files, for the settings the notebooks read.
5. `src/README.md`, then `src/ecg/README.md`, then the subpackage READMEs in the order they list.
6. `results/README.md`, `max78002/README.md` and `report/README.md`, to see where the outputs go.

### How the pieces connect

- `notebooks/EDA.ipynb` loads `configs/preprocess.yaml`, downloads `data/raw/`, calls `src/ecg/data/pipeline.py` (writes `data/processed/` and the tables in `results/`), and draws the EDA figures for report section 2.
- `notebooks/MODELING.ipynb` loads `configs/model.yaml`. It trains the baselines and the CNN sweep (`src/ecg/training/`), quantises with the toolchain (`src/ecg/deployment/`), scores DS2 and runs the noise test (`src/ecg/evaluation/`, `src/ecg/noise/`), synthesises the C project into `max78002/`, and writes every table and figure of report sections 4 to 6 into `results/`.
- Both notebooks copy their figures and tables into `report/`, run the tests, and build `report.pdf`.
- The report sections quote only numbers that exist in `results/` or `max78002/`.

## 4. Installation

Python 3.10 or 3.11. `uv` is used because `python -m venv` may lack `ensurepip` on WSL.

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name d7043e --display-name "Python (D7043E)"
make toolchain
```

Main dependencies: `wfdb`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `torch`, `pyyaml`, `jupyter`. `requirements.txt` pins the CUDA build of PyTorch that was used; the CPU build works too (`uv pip install torch --index-url https://download.pytorch.org/whl/cpu`), training then takes about ten times longer.

`make toolchain` clones `ai8x-training` and `ai8x-synthesis` from Analog Devices into `toolchain/` and creates the separate Python 3.11 environment the synthesis scripts need. The notebook imports the ai8x layer wrappers from the first repository and runs `quantize.py` and `ai8xize.py` from the second.

## 5. Running everything

Two notebooks are the only entry points. Open them with the `.venv` kernel and choose *Restart kernel and run all*, or run them headless:

```bash
make run        # = make eda, then make model
```

`EDA.ipynb` (Submission 1) runs these steps:

| Section | What it does | Writes |
|---------|--------------|--------|
| 0.1 | Downloads `mitdb` and `nstdb` from PhysioNet, if `data/raw/` is empty | `data/raw/mitdb/`, `data/raw/nstdb/` |
| 0.2 | Preprocessing: filter, cut, normalise, split (skipped if the output exists) | `data/processed/{train,val,test}.npz`, `results/split_stats.csv`, `results/record_stats.csv`, `results/label_mapping.json` |
| 2.1 to 2.6 | EDA for report section 2 | `results/eda/` |
| 3 to 5 | Copies into the report, tests, `report.pdf` | |

`MODELING.ipynb` (Submission 2) runs these steps:

| Section | What it does | Writes |
|---------|--------------|--------|
| 1 | Baselines: logistic regression and random forest on validation | `results/experiments.csv`, `results/checkpoints/baseline_*.joblib` |
| 2 | Model-selection sweep of `configs/sweep.yaml` on validation; builds `data/processed_w512/` for the wide-window run | `results/experiments.csv`, `results/history/`, `results/checkpoints/*_fp32.pth.tar` |
| 3 | The best run becomes the final CNN: layer table, training settings, learning curve | `results/model_layers.csv`, `results/tables/`, `results/figures/` |
| 4 | Quantisation: PTQ with `quantize.py`, QAT fine-tuning, `quantize.py` again; INT8 model chosen on validation | `max78002/quantization_output/`, `results/quantization_val.csv` |
| 5 | DS2 evaluation, once per model | `results/metrics.csv`, `results/confusion_matrix*.{csv,png}` |
| 6 | Noise test at 12, 6 and 0 dB for three noise types | `results/noise_results.{csv,png}` |
| 7 | Error analysis | `results/figures/error_examples*.png` |
| 8 | Synthesis with `ai8xize.py`, evidence table | `max78002/model_configuration/`, `max78002/synthesis_output/`, `results/deployment_evidence.csv` |
| 9 | Success-criteria review | `results/success_criteria.csv` |
| 10 | Copies into the report, tests, `report.pdf` | `report/figures/`, `report/tables/`, `report.pdf` |

Runs already listed in `results/experiments.csv` are skipped, so the notebook can be rerun quickly. Set `FORCE_TRAIN = True` in its first cell to train everything again. A full run from scratch takes about 40 minutes on a laptop GPU.

Raw data is never committed. If the download fails with an SSL certificate error (common behind a proxy or in WSL), the notebook already points Python at the system certificate store, `/etc/ssl/certs/ca-certificates.crt`. To download by hand instead:

```bash
mkdir -p data/raw
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"
```

## 6. Shortcuts

```bash
make eda        # EDA.ipynb only
make model      # MODELING.ipynb only
make test       # pytest
make report     # pdflatex, biber, pdflatex x2, copies report.pdf to the project root
```

### Preprocessing steps

The random seed and all preprocessing parameters are in `configs/preprocess.yaml`.

1. Read the MLII lead of each record at 360 Hz.
2. Remove baseline wander with two median filters, then remove fast noise with a 40 Hz low-pass filter.
3. Cut a fixed window around each annotated R-peak: 256 samples in Submission 1, 512 samples (`configs/preprocess_w512.yaml`) for the final model, chosen by the sweep.
4. Normalise each beat and clip to [-8, 8].
5. Compute the RR-interval features of each beat (used by the baselines only).

The model divides the clipped beat by 8, which gives the accelerator input range [-1, 1).

## 7. Edge-AI constraints (MAX78002)

- Only accelerator operations: fused `MaxPool1d` + `Conv1d` + `ReLU` blocks and one `Linear` output layer, built with the `ai8x` wrappers.
- Fixed-size input (1 x 512), 25 348 weights, about 25 kB in INT8, far under the 2 MB limit.
- Checked by `ai8x`'s own layer asserts, by `quantize.py` and by `ai8xize.py`, which generated the C project in `max78002/synthesis_output/`.
- The Python INT8 simulation and the synthesiser's known-answer output agree bit for bit on the sample beat.

## 8. Group

| Name | Programme | Role |
|------|-----------|------|
| Ekpenyong Esu Ekpenyong | | |
| Sreenidhi Thallam | | |
| Sowmya Nelakurthi | | |

## 9. References

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol, 2001.
- Moody GB, Muldrow WE, Mark RG. *A noise stress test for arrhythmia detectors.* Computers in Cardiology, 1984.
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE TBME, 2004. (DS1/DS2 protocol)
- Analog Devices, ai8x-training: https://github.com/analogdevicesinc/ai8x-training
- Analog Devices, ai8x-synthesis: https://github.com/analogdevicesinc/ai8x-synthesis
- Analog Devices, MAX78002 documentation: https://www.analog.com/en/products/max78002.html
- WFDB Python package: https://github.com/MIT-LCP/wfdb-python
