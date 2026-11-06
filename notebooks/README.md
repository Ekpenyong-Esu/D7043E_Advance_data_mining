# notebooks/

Start with `EDA.ipynb`, then `MODELING.ipynb`. They are the only entry points of the project.

1. `EDA.ipynb` (Submission 1): download, preprocessing, EDA, report sections 1 to 3. *Restart kernel and run all*, or `make eda`.
2. `MODELING.ipynb` (Submission 2): baselines, model selection, the CNN, quantisation, DS2 evaluation, noise test, error analysis, MAX78002 synthesis, report sections 4 to 8. *Restart kernel and run all*, or `make model`.

Sections of `EDA.ipynb`, in order:

0. Setup. Loads `configs/preprocess.yaml`, downloads `mitdb` and `nstdb` into `data/raw/` if they are missing, runs the preprocessing (`ecg.data.pipeline`) into `data/processed/` and `results/`.
1. EDA, report section 2. Calls `stats` and `plots` from `src/ecg/eda/`. Writes `results/eda/`.
2. Report. Copies the figures and tables into `report/`, runs `pytest`, builds `report.pdf`.

Sections of `MODELING.ipynb`, in order:

0. Setup. Loads `configs/model.yaml` and imports the ai8x layers from `toolchain/`.
1. Baselines (`ecg.training.runner.run_baseline`).
2. Model selection (`ecg.training.sweep.run_sweep`), validation only.
3. The final CNN: layer table, settings, learning curve.
4. Quantisation: PTQ, QAT, `quantize.py` (`ecg.deployment.toolchain`), INT8 chosen on validation.
5. DS2 evaluation (`ecg.evaluation`), once per model.
6. Noise test (`ecg.noise.experiment`).
7. Error analysis.
8. Deployment: network YAML, sample beat, `ai8xize.py`, evidence table.
9. Success criteria.
10. Report: copy, tests, `report.pdf`.

How they connect: the notebooks define no logic of their own. Every step is one call into `src/ecg/`. They are the only code that draws figures. Flags at the top of each first cell control the optional steps (`FORCE_DOWNLOAD`, `FORCE_PREPARE`, `FORCE_TRAIN`, `RUN_TESTS`, `BUILD_REPORT`). `MODELING.ipynb` needs `data/processed/` from `EDA.ipynb`. Run both with the `.venv` kernel.
