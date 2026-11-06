# How to build the project with the Makefile

## Understanding the problem

The project teaches a small model to look at one heartbeat and decide which of four types it is. These are the standard AAMI classes:

| Class | Name | What happens in the heart |
|---|---|---|
| **N** | Normal beat | The beat starts in the SA node (the heart's natural pacemaker) and travels the normal path. Some harmless variations, such as bundle branch blocks, are counted here too. |
| **S** | Supraventricular ectopic beat | An early beat that starts in the upper part of the heart (the atria), but not at the SA node. It looks almost normal, but it comes too early. |
| **V** | Ventricular ectopic beat | An early beat that starts in the lower chambers (the ventricles). It skips the normal path, so the QRS is wide and oddly shaped. This is the most important one to catch. |
| **F** | Fusion beat | A normal beat and a V beat happen at the same moment and merge into one. Its shape is halfway between N and V. It is very rare. |

Paced beats (from a pacemaker) and unknown beats are left out.

---

Everything runs from two notebooks. `notebooks/EDA.ipynb` is Submission 1 (the data). `notebooks/MODELING.ipynb` is Submission 2 (the model). Open them with the `.venv` kernel and choose *Restart kernel and run all*. Or run both from the project folder without opening them:

```bash
make run
```

Before the first run, the `.venv` folder must exist, and `make toolchain` must have cloned the Analog Devices repositories.

---

## Part A: `EDA.ipynb` (make eda)

### 1. Section 0.1: download the data

**What it does:** downloads the two ECG databases into `data/raw/`.
- `mitdb`: the heart recordings, with every beat labelled.
- `nstdb`: real ECG noise.

It first points Python at the system certificates, so the download does not fail behind a company network. It is skipped when the files are already there. Set `FORCE_DOWNLOAD = True` in the first cell to download again.

**Why first:** every other step needs these files.

### 2. Section 0.2: prepare the data

**What it does:** turns the raw recordings into clean heartbeats, using the settings in `configs/preprocess.yaml`. Results go to `data/processed/` (train, val and test) and `results/`. The work is done by `src/ecg/data/pipeline.py`; the notebook only calls it.

It is skipped when `data/processed/` already exists. Set `FORCE_PREPARE = True` to rerun it, for example after changing the config.

**Why second:** it needs the raw data from step 1, and the next steps need its output.

### 3. Sections 2.1 to 2.6: EDA

**What it does:** makes the figures and tables that describe the data. They go to `results/eda/`. Section 3 of the notebook then copies them into `report/figures/` and `report/tables/`.

### 4. Section 4: run the tests, Section 5: build the report

Same as steps 10 and 11 of Part B below.

---

## Part B: `MODELING.ipynb` (make model)

### 1. Section 1: baselines

**What it does:** trains a logistic regression and a random forest on the flattened beat plus the three RR-interval features (`configs/baselines.yaml`), and scores them on the validation records. One row each goes to `results/experiments.csv`.

**Why first:** the CNN needs a simple reference point. The RR features tell the baseline how early a beat came, which the CNN has to learn from the waveform.

### 2. Section 2: model selection

**What it does:** runs the sweep in `configs/sweep.yaml`. Stage 1 tries ways to handle the class imbalance. Stage 2 varies the architecture, regularisation, augmentation and the beat window. Stage 3 varies the rest again around the stage-2 winner. Each stage starts from the best run so far. The score is macro-F1 on the validation records. Runs already in `results/experiments.csv` are skipped.

**Why here:** every later step uses the winner. DS2 is not touched.

### 3. Section 3: the final CNN

**What it does:** rebuilds the config of the winning run and writes its layer table, training settings and learning curve for report sections 4.2 and 4.3.

### 4. Section 4: quantisation

**What it does:** makes two INT8 models with `quantize.py` from ai8x-synthesis. The first comes straight from the FP32 weights (post-training quantisation). The second comes after a short quantisation-aware fine-tuning, where the ai8x layers round the weights to 8 bits and clamp the activations during training. Both are scored on the validation records in the ai8x integer simulation. The better one is the deployed model.

**Why before DS2:** the FP32-versus-INT8 decision is part of model selection, so it must use validation, not the test set.

### 5. Section 5: evaluation on DS2

**What it does:** reads the test split for the first time (`final=True`) and scores the baselines, the FP32 CNN and the INT8 CNN once each. Writes `results/metrics.csv` and the confusion matrices.

**Why now:** all models are frozen. Nothing is changed after this.

### 6. Section 6: noise robustness

**What it does:** adds `bw`, `em` and `ma` noise from `nstdb` to the raw DS2 signals at 12, 6 and 0 dB, runs the same preprocessing, and scores the frozen models. Also runs the small and large CNNs from the sweep (robustness versus size) and the INT8 model with the low-pass filter off (does the preprocessing help). Writes `results/noise_results.csv` and the figure.

### 7. Section 7: error analysis

**What it does:** draws misclassified beats (N as S, S as N, V wrong, F wrong) and beats that were right when clean but wrong at 0 dB.

### 8. Section 8: deployment

**What it does:** writes the network YAML, exports one sample beat, and runs `ai8xize.py`, which generates the C project for the MAX78002 in `max78002/synthesis_output/`. The synthesiser's expected output for the sample beat is compared with the Python INT8 simulation. The PASS/FAIL table goes to `results/deployment_evidence.csv`.

### 9. Section 9: success criteria

**What it does:** checks every criterion of report section 1.4 against the DS2 results and writes `results/success_criteria.csv`.

### 10. Section 10: tests

**What it does:** runs the tests in `tests/` with pytest. The notebook stops if a test fails, so a report is never built from broken code.

```bash
make test
```

### 11. Section 10: build the report

**What it does:** builds the report PDF and copies it to `report.pdf` in the project folder.
- `pdflatex` turns the LaTeX into a PDF.
- `biber` adds the references.
- `pdflatex` runs twice more so the page numbers, table of contents and citations come out right.

It is skipped when `pdflatex` or `biber` is not installed, or when `BUILD_REPORT = False`.

```bash
make report
```

**Why last:** the report uses the figures and tables made in the steps before.
