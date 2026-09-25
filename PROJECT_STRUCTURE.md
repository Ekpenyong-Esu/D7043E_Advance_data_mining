# Project Structure

The assignment requires six scripts in `src/`. To keep every file short and
single-purpose, those six are **thin entry points** and the real logic lives
in the `src/ecg/` package, one module per responsibility.

```
D7043E/                               (GroupXX_ECG_Project)
│
├── README.md                         Assignment text (verbatim)
├── PROJECT_README.md                 Submission README: install, download, reproduce
├── STEP_BY_STEP.md                   Execution guide mapped to CRISP-DM
├── APPROACHES.md                     Review of the guide and alternative approaches
├── PROJECT_STRUCTURE.md              Responsibility of every file, design rules
├── requirements.txt                  Python dependencies (pip install -r)
├── pyproject.toml                    pytest / ruff settings
├── Makefile                          make data | prepare | train | qat | evaluate | noise | test
├── .gitignore                        keeps data/, checkpoints, caches out of git
│
├── configs/                          All tunable numbers; code has no magic values
│   ├── preprocess.yaml               lead, window, filters, normalization, record lists
│   ├── baseline.yaml                 sklearn baseline
│   ├── cnn_fp32.yaml                 CNN, FP32 training
│   ├── cnn_qat.yaml                  CNN, quantization-aware training
│   └── noise.yaml                    noise-robustness experiment
│
├── data/                             git-ignored, never submitted
│   ├── raw/                          mitdb/ and nstdb/ downloaded with wfdb
│   └── processed/                    train.npz, val.npz, test.npz
│
├── notebooks/                        Jupyter notebooks (analysis only, no pipeline code)
│   └── EDA.ipynb                     Report §2; uses ecg.data helpers, defines no logic
│
├── src/                              All source code
│   ├── prepare_data.py               CLI  →  ecg.data.pipeline
│   ├── dataset.py                    re-export of ecg.datasets
│   ├── model.py                      re-export of ecg.models
│   ├── train.py                      CLI  →  ecg.training.runner
│   ├── evaluate.py                   CLI  →  ecg.evaluation.runner  (test split needs --final)
│   ├── noise_test.py                 CLI  →  ecg.noise.experiment
│   ├── make_eda_figures.py           CLI  →  ecg.eda (figures + tables for report §2)
│   ├── fill_report_numbers.py        CLI: writes results/*.csv numbers into report/sections
│   │
│   └── ecg/                          The package: one responsibility per file
│       ├── __init__.py               package marker
│       ├── constants.py              DS1 / DS2, class names, symbol → class map, exclusions
│       ├── config.py                 YAML loading
│       ├── seed.py                   determinism
│       ├── logging_utils.py          logger factory
│       │
│       ├── eda/                      CRISP-DM 2: Data Understanding
│       │   ├── style.py              class colours, figure style
│       │   ├── stats.py              per-record / per-split statistics
│       │   └── plots.py              the six report figures
│       │
│       ├── data/                     CRISP-DM 3: Data Preparation
│       │   ├── download.py           PhysioNet download
│       │   ├── records.py            read one record + annotations
│       │   ├── channels.py           select lead by name (record 114 is swapped)
│       │   ├── filters.py            baseline-wander / low-pass filtering
│       │   ├── beats.py              fixed windows around R-peaks
│       │   ├── labels.py             symbol → N/S/V/F, exclusion counts
│       │   ├── normalize.py          per-beat normalization + clipping
│       │   ├── rr.py                 pre/post RR intervals (timing cue for S beats)
│       │   ├── splits.py             record-level split, disjointness check
│       │   ├── storage.py            npz IO, split statistics table
│       │   └── pipeline.py           wires the stages; the only orchestrator
│       │
│       ├── datasets/                 torch data access
│       │   ├── beat_dataset.py       Dataset wrapper
│       │   ├── augment.py            training-only augmentations
│       │   ├── samplers.py           minority oversampling
│       │   └── loaders.py            DataLoaders from config
│       │
│       ├── models/                   CRISP-DM 4: Modeling
│       │   ├── baseline.py           Random Forest / Logistic Regression
│       │   ├── cnn.py                ECGCNN: Conv1d, pooling, ReLU, Linear only
│       │   ├── ai8x_cnn.py           same network with ai8x wrappers
│       │   └── registry.py           config → model
│       │
│       ├── training/                 training loop and its helpers
│       │   ├── losses.py             class weights, focal loss
│       │   ├── early_stopping.py     stop when validation metric stalls
│       │   ├── checkpoint.py         save / load weights + config
│       │   ├── loop.py               one epoch train / predict, no IO
│       │   ├── experiment_log.py     results/experiments.csv
│       │   └── runner.py             wires one training run
│       │
│       ├── evaluation/               CRISP-DM 5: Evaluation
│       │   ├── metrics.py            accuracy, macro-F1, per-class P / R / F1
│       │   ├── confusion.py          confusion matrix csv + png
│       │   ├── report.py             results/metrics.csv
│       │   ├── errors.py             representative misclassifications (§5.5)
│       │   └── runner.py             wires one evaluation
│       │
│       ├── noise/                    robustness experiment (§5.4)
│       │   ├── nstdb.py              load bw / em / ma
│       │   ├── snr.py                add noise at a target SNR
│       │   └── experiment.py         clean / 12 dB / 6 dB / 0 dB
│       │
│       └── deployment/               CRISP-DM 6: Deployment
│           ├── sample_input.py       .npy for ai8xize --sample-input
│           └── evidence.py           PASS / FAIL table from toolchain logs
│
├── tests/                            pytest; pure functions only
│   ├── conftest.py                   shared pytest fixtures (rng)
│   ├── test_labels.py                symbol → class mapping and exclusions
│   ├── test_splits.py                DS1/DS2 split is disjoint and complete
│   ├── test_beats.py                 window length and boundary handling
│   ├── test_rr.py                    premature beat has short pre-RR, long post-RR
│   ├── test_filters.py               baseline removal and low-pass behave as intended
│   ├── test_normalize.py             z-score and clipping
│   └── test_snr.py                   noise is added at the requested SNR
│
├── results/                          metrics.csv, confusion_matrix.*, noise_results.*, split_stats.csv
│   ├── checkpoints/                  trained weights (*.pth, *.pkl; git-ignored)
│   └── figures/                      plots produced by evaluate.py / noise_test.py
│
├── max78002/                         deployment evidence for report §6
│   ├── model_configuration/          network YAML + ai8x model / dataset files
│   ├── quantization_output/          quantize.py log + -q.pth.tar
│   └── synthesis_output/             ai8xize.py output: C code, weights.h, log
│
└── report/                           LaTeX report sources → report.pdf at root
    ├── report.tex                    main file (D7043E, Group 16)
    ├── thesisMDH.cls                 LTU/MDH document class
    ├── references.bib
    ├── README.md                     how the report sources are organised and built
    ├── sections/                     01_business_understanding … 08_limitations, one .tex per CRISP-DM section
    ├── figures/                      figures exported for the report
    └── images/                       university logos used by the template
```

## Design rules

1. **Entry points do nothing but parse arguments and call one function.**
2. **`constants.py` is the only place record lists and label maps are written.**
3. **Configs hold every tunable number.** A module never hard-codes a window size or a cut-off.
4. **Pure functions in `data/`, `evaluation/`, `noise/` take arrays and return arrays.** File IO is confined to `storage.py`, `checkpoint.py`, `report.py` and the runners.
5. **Runners orchestrate; they contain no algorithms.** `pipeline.py`, `training/runner.py`, `evaluation/runner.py`, `noise/experiment.py`.
6. **One model definition.** `ai8x_cnn.py` mirrors `cnn.py` layer for layer; change both or neither.
7. **Test-set discipline is enforced in code.** `evaluate.py --split test` fails without `--final`.
8. **Target file length:** under ~80 lines. If a module grows past that, split it.

## Status

**Submission 1 complete.** `make data`, `make prepare`, `make eda` and `make test` run end to end.
Implemented: constants, config, seed, logging, records, channels, filters, beats, normalize, rr,
labels, splits, storage, pipeline, the `eda` package, dataset wrapper, baseline models, metrics,
early stopping, checkpoint IO, experiment log, NSTDB loading, all entry points, 9 tests.

**Submission 2 stubs** (`raise NotImplementedError`), in the order the guide needs them:
`datasets/augment.py`, `samplers.py`, `loaders.py` → `models/cnn.py`, `training/losses.py`,
`loop.py`, `training/runner.py` → `evaluation/confusion.py`, `report.py`, `errors.py`,
`evaluation/runner.py` → `noise/snr.py`, `noise/experiment.py` → `models/ai8x_cnn.py`,
`deployment/sample_input.py`. The ai8x network YAML in `max78002/model_configuration/` is written by hand.
