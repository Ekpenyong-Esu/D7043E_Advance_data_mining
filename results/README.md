# results/

Files written by the notebooks. The report quotes its numbers from here.

Start with `split_stats.csv`. Read in this order:

1. `split_stats.csv`: beats per class per split (report table 3.8).
2. `record_stats.csv`: the same per record, plus edge-dropped and excluded counts.
3. `label_mapping.json`: the symbol to class mapping used.
4. `eda/`: tables and figures for report section 2 (Submission 1).
5. `experiments.csv`: one row per training run (baselines, every sweep run, QAT): validation scores, settings, and the config overrides that produced it (report section 4.4).
6. `history/`: per-epoch training loss and validation scores of every CNN run.
7. `checkpoints/`: the trained models.
8. `model_layers.csv`: the layer table of the final CNN (section 4.2).
9. `quantization_val.csv`: FP32, PTQ-INT8 and QAT-INT8 on validation (section 4.5).
10. `metrics.csv`: DS2 scores of the baselines, the FP32 CNN and the INT8 CNN (sections 5.1 to 5.3).
11. `confusion_matrix.csv` and `confusion_matrix.png`: the deployed INT8 model on DS2; `confusion_matrix_<model>.*` for the others.
12. `noise_results.csv` and `noise_results.png`: the noise experiment (section 5.4).
13. `deployment_evidence.csv`: the PASS/FAIL table of section 6.3.
14. `success_criteria.csv`: the review of section 5.6.
15. `figures/` and `tables/`: every figure and LaTeX table of report sections 4 to 6, copied into `report/`.
16. `w512/`: split statistics of the 512-sample window variant.

How they connect:

- Section 0.2 of `notebooks/EDA.ipynb` calls `src/ecg/data/pipeline.py`, which writes the first three files. Sections 2.1 to 2.6 write `eda/`.
- `notebooks/MODELING.ipynb` writes everything else, in the order of its sections, through `src/ecg/training/`, `src/ecg/evaluation/`, `src/ecg/noise/` and `src/ecg/deployment/`.
- `report/sections/04` to `08` include the tables in `tables/` and the figures in `figures/`, and quote `metrics.csv`, `noise_results.csv` and `deployment_evidence.csv`.
