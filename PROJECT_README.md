# ECG Heartbeat Classification for Edge AI (MAX78002)

Course project for **D7043E** at Luleå University of Technology (LTU).
A compact 1D-CNN that classifies ECG heartbeats from the MIT-BIH Arrhythmia
Database into four classes (**N / S / V / F**), developed with the
**CRISP-DM** methodology and prepared for software-level deployment on the
**Analog Devices MAX78002** Edge-AI microcontroller.

> This is a course project. The model is **not** a medical device and is
> **not** clinically validated. No physical MAX78002 board is used, so no
> hardware measurements (latency, power, energy, on-device accuracy) are claimed.

---

## 1. Project goal

Classify single ECG heartbeats into clinically meaningful categories while
respecting Edge-AI constraints (model size ≤ 2 MB, INT8 inference, operations
supported by the MAX78002 CNN accelerator).

| Class | Meaning                          |
|-------|----------------------------------|
| N     | Normal-type beats                |
| S     | Supraventricular ectopic beats   |
| V     | Ventricular ectopic beats        |
| F     | Fusion beats                     |

Rare paced / unclassifiable beats (e.g. `/`, `f`, `Q`) are excluded and the
exclusion is documented in the report.

---

## 2. Datasets

| Dataset | Use | Source |
|---------|-----|--------|
| MIT-BIH Arrhythmia Database (`mitdb`) | Main training / validation / test data | https://physionet.org/content/mitdb/1.0.0/ |
| MIT-BIH Noise Stress Test Database (`nstdb`) | Noise-robustness experiment | https://physionet.org/content/nstdb/1.0.0/ |

Key facts (mitdb): 48 records, 47 subjects, 2 channels, 360 Hz, ~30 min each,
~110 000 expert beat annotations.

**Raw data is not committed to this repository.** See
[Section 5](#5-downloading-the-data) for download instructions.

### Inter-patient split (de Chazal protocol)

Beats are **never** split randomly. Records are assigned whole to one split.

| Split | Records |
|-------|---------|
| **DS1** (development: train + validation, split *by record*) | 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230 |
| **DS2** (final test, untouched until final evaluation) | 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234 |

The exact train / validation partition of DS1 is defined in `configs/`.

---

## 3. Repository structure

```
GroupXX_ECG_Project/
│
├── report.pdf                # CRISP-DM report (Submission 1 + 2)
├── README.md                 # This file
├── STEP_BY_STEP.md           # Detailed walkthrough of the whole workflow
├── requirements.txt
│
├── notebooks/
│   └── EDA.ipynb             # Exploratory data analysis
│
├── src/
│   ├── prepare_data.py       # Download check, beat extraction, preprocessing, splits
│   ├── dataset.py            # PyTorch Dataset / DataLoader, augmentation
│   ├── model.py              # Baseline + compact 1D-CNN (ai8x-compatible)
│   ├── train.py              # Training loop (FP32 and QAT)
│   ├── evaluate.py           # Metrics, confusion matrix, per-class results
│   └── noise_test.py         # Noise-robustness evaluation (12 / 6 / 0 dB)
│
├── configs/                  # YAML configs: preprocessing, model, training
│
├── results/
│   ├── metrics.csv
│   ├── confusion_matrix.*
│   └── noise_results.*
│
└── max78002/
    ├── model_configuration   # ai8x network description + YAML
    ├── quantization_output   # quantize.py output (INT8 checkpoint + log)
    └── synthesis_output      # ai8xize.py output (generated C code + log)
```

---

## 4. Installation

```bash
# Python 3.10 or 3.11 recommended
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Main dependencies: `wfdb`, `numpy`, `scipy`, `pandas`, `scikit-learn`,
`matplotlib`, `torch`, `pyyaml`, `jupyter`.

The MAX78002 toolchain (`ai8x-training`, `ai8x-synthesis`) is installed
separately. See [STEP_BY_STEP.md](STEP_BY_STEP.md), Step 9.

---

## 5. Downloading the data

Do **not** commit the raw datasets. Download them into `data/raw/`:

```bash
mkdir -p data/raw
# Option A: wfdb helper (recommended)
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"

# Option B: wget
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/mitdb https://physionet.org/files/mitdb/1.0.0/
wget -r -N -c -np -nH --cut-dirs=3 -P data/raw/nstdb https://physionet.org/files/nstdb/1.0.0/
```

Expected layout afterwards:

```
data/raw/mitdb/100.dat  100.hea  100.atr  ...
data/raw/nstdb/bw.dat   em.dat   ma.dat   ...
```

---

## 6. Reproducing the pipeline

All steps are deterministic given the seed in `configs/`.

```bash
# 1. Preprocess + build splits (writes data/processed/*.npz and results/split_stats.csv)
python src/prepare_data.py --config configs/preprocess.yaml

# 2. Exploratory data analysis
jupyter notebook notebooks/EDA.ipynb

# 3. Baseline (e.g. logistic regression / random forest)
python src/train.py --config configs/baseline.yaml

# 4. Compact 1D-CNN, FP32
python src/train.py --config configs/cnn_fp32.yaml

# 5. Quantization-aware training (INT8)
python src/train.py --config configs/cnn_qat.yaml

# 6. Evaluate on DS2 (only after model selection is frozen)
python src/evaluate.py --checkpoint results/checkpoints/cnn_qat_best.pth --split test

# 7. Noise robustness (clean / 12 dB / 6 dB / 0 dB)
python src/noise_test.py --checkpoint results/checkpoints/cnn_qat_best.pth
```

Outputs are written to `results/`.

### Preprocessing summary

```
Raw ECG (360 Hz, MLII)
  → band-pass / baseline-wander removal
  → beat segmentation around annotated R-peak (fixed window)
  → per-beat normalization
  → clipping
  → fixed-length model input
```

Exact parameters (window size, filter cut-offs, normalization method, seed)
live in `configs/preprocess.yaml` and are reported in the CRISP-DM report,
Section 3.9.

---

## 7. Edge-AI constraints (MAX78002)

* Model size ≤ 2 MB, INT8 weights.
* Only accelerator-supported ops: `Conv1d`, `MaxPool1d` / `AvgPool1d`,
  `ReLU`, small `Linear` output layer, fixed-size input.
* Architecture is defined with `ai8x` layer wrappers so it can be quantized
  with `quantize.py` and synthesized with `ai8xize.py`.
* No Transformers, LSTMs/GRUs, large ResNets or unsupported custom ops.

Deployment evidence (toolchain acceptance, quantization, synthesis, generated
C files, weight / data memory requirements) is stored under `max78002/` and
reported in the CRISP-DM report, Section 6.

---

## 8. Rules followed

- [x] No random mixing of beats from the same record across splits.
- [x] DS2 never used for model selection or hyperparameter tuning.
- [x] Noise experiment never leaks test information into training.
- [x] Accuracy is never reported alone (macro-F1, per-class precision / recall / F1, confusion matrix).
- [x] No clinical-validation claims.
- [x] No hardware measurements claimed without a physical board.
- [x] All preprocessing, exclusions and class mappings are reproducible.
- [x] All six CRISP-DM stages documented.

---

## 9. Group

| Name | Programme | Role |
|------|-----------|------|
|      |           |      |

Fill in before submission. Max 5 members; PhD students work alone.

---

## 10. References

* Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng Med Biol, 2001.
* Moody GB, Muldrow WE, Mark RG. *A noise stress test for arrhythmia detectors.* Computers in Cardiology, 1984.
* de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE TBME, 2004. (DS1/DS2 protocol)
* Analog Devices, *ai8x-training*: https://github.com/analogdevicesinc/ai8x-training
* Analog Devices, *ai8x-synthesis*: https://github.com/analogdevicesinc/ai8x-synthesis
* Analog Devices, MAX78002 documentation: https://www.analog.com/en/products/max78002.html
* WFDB Python package: https://github.com/MIT-LCP/wfdb-python
