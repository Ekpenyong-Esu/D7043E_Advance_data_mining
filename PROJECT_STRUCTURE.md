# Project Structure

The assignment requires six scripts in `src/`. To keep every file short and
single-purpose, those six are **thin entry points** and the real logic lives
in the `src/ecg/` package, one module per responsibility.

```
D7043E/  (GroupXX_ECG_Project)
│
├── README.md                 # Assignment text (verbatim)
├── PROJECT_README.md         # Submission README: install, download, reproduce
├── STEP_BY_STEP.md           # Execution guide mapped to CRISP-DM
├── APPROACHES.md             # Review of the guide + alternative approaches
├── PROJECT_STRUCTURE.md      # This file
├── requirements.txt
├── pyproject.toml            # pytest / ruff settings
├── Makefile                  # make data | prepare | train | qat | evaluate | noise | test
├── .gitignore
│
├── configs/                  # All tunable numbers. Code contains no magic values.
│   ├── preprocess.yaml       #   lead, window, filters, normalization, record lists
│   ├── baseline.yaml         #   sklearn baseline
│   ├── cnn_fp32.yaml         #   CNN, FP32 training
│   ├── cnn_qat.yaml          #   CNN, quantization-aware training
│   └── noise.yaml            #   noise-robustness experiment
│
├── data/                     # git-ignored
│   ├── raw/                  #   mitdb/, nstdb/ downloaded by wfdb
│   └── processed/            #   train.npz, val.npz, test.npz
│
├── notebooks/
│   └── EDA.ipynb             # Report §2; imports helpers from ecg.data, defines no logic
│
├── src/
│   ├── prepare_data.py       # CLI → ecg.data.pipeline
│   ├── dataset.py            # re-exports ecg.datasets
│   ├── model.py              # re-exports ecg.models
│   ├── train.py              # CLI → ecg.training.runner
│   ├── evaluate.py           # CLI → ecg.evaluation.runner (refuses test split without --final)
│   ├── noise_test.py         # CLI → ecg.noise.experiment
│   │
│   └── ecg/                  # The package. One responsibility per file.
│       ├── constants.py      #   DS1/DS2, class names, symbol→class map, exclusions
│       ├── config.py         #   YAML loading
│       ├── seed.py           #   determinism
│       ├── logging_utils.py  #   logger factory
│       │
│       ├── data/             #   CRISP-DM stage 3
│       │   ├── download.py   #     PhysioNet download
│       │   ├── records.py    #     read one record + annotations
│       │   ├── channels.py   #     pick lead by name (record 114 trap)
│       │   ├── filters.py    #     baseline / low-pass filtering
│       │   ├── beats.py      #     fixed windows around R-peaks
│       │   ├── labels.py     #     symbol → N/S/V/F, exclusion counts
│       │   ├── normalize.py  #     per-beat normalization + clipping
│       │   ├── splits.py     #     record-level split, disjointness check
│       │   ├── storage.py    #     npz IO, split statistics table
│       │   └── pipeline.py   #     wires the above; the only orchestrator
│       │
│       ├── datasets/         #   torch data access
│       │   ├── beat_dataset.py
│       │   ├── augment.py    #     training-only augmentations
│       │   ├── samplers.py   #     minority oversampling
│       │   └── loaders.py    #     build DataLoaders from config
│       │
│       ├── models/           #   CRISP-DM stage 4
│       │   ├── baseline.py   #     RF / LR
│       │   ├── cnn.py        #     ECGCNN (Conv1d / pool / ReLU / Linear only)
│       │   ├── ai8x_cnn.py   #     same net with ai8x wrappers
│       │   └── registry.py   #     config → model
│       │
│       ├── training/
│       │   ├── losses.py     #     class weights, focal loss
│       │   ├── early_stopping.py
│       │   ├── checkpoint.py
│       │   ├── loop.py       #     one epoch train / predict, no IO
│       │   ├── experiment_log.py  # results/experiments.csv
│       │   └── runner.py     #     wires a run
│       │
│       ├── evaluation/       #   CRISP-DM stage 5
│       │   ├── metrics.py    #     accuracy, macro-F1, per-class
│       │   ├── confusion.py
│       │   ├── report.py     #     results/metrics.csv
│       │   ├── errors.py     #     representative misclassifications (§5.5)
│       │   └── runner.py
│       │
│       ├── noise/            #   robustness experiment (§5.4)
│       │   ├── nstdb.py      #     load bw / em / ma
│       │   ├── snr.py        #     add noise at target SNR
│       │   └── experiment.py
│       │
│       └── deployment/       #   CRISP-DM stage 6
│           ├── sample_input.py   # .npy for ai8xize --sample-input
│           ├── network_yaml.py   # ai8x network description
│           └── evidence.py       # PASS/FAIL table from toolchain logs
│
├── tests/                    # pytest; pure functions are tested, IO is not
│   ├── test_labels.py
│   ├── test_splits.py
│   ├── test_beats.py
│   └── test_snr.py
│
├── results/                  # metrics.csv, confusion_matrix.*, noise_results.*, split_stats.csv
│   ├── checkpoints/
│   └── figures/
│
├── max78002/
│   ├── model_configuration/  # network YAML + ai8x model/dataset files
│   ├── quantization_output/  # quantize.py log + -q.pth.tar
│   └── synthesis_output/     # ai8xize.py output (C code, weights.h, log)
│
└── report/                   # report sources, one file per section → report.pdf at root
    └── figures/
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

Fully implemented: constants, config, seed, logging, records, channels, labels,
splits, storage, pipeline wiring, dataset wrapper, baseline models, metrics,
early stopping, checkpoint IO, experiment log, NSTDB loading, entry points.

Stubs (`raise NotImplementedError`) to fill in, in the order the guide needs them:
`filters.py`, `beats.py`, `normalize.py` → `augment.py`, `samplers.py`, `loaders.py`
→ `cnn.py`, `losses.py`, `loop.py`, `training/runner.py` → `confusion.py`,
`report.py`, `errors.py`, `evaluation/runner.py` → `snr.py`, `noise/experiment.py`
→ `ai8x_cnn.py`, `deployment/*`.
