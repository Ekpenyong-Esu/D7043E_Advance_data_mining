# results/

Files written by the pipeline. The report quotes its numbers from here.

Start with `split_stats.csv`. Read in this order:

1. `split_stats.csv`: beats per class per split (report table 3.8).
2. `record_stats.csv`: the same per record, plus edge-dropped and excluded counts.
3. `label_mapping.json`: the symbol to class mapping used.
4. `eda/`: tables for report section 2.
5. `figures/`: plots from evaluation and the noise test (Submission 2).
6. `checkpoints/`: trained weights, git-ignored.

How they connect:

- `make prepare` (`src/prepare_data.py`) writes the first three files at the end of `src/ecg/data/pipeline.py`.
- `make eda` (`src/make_eda_figures.py`) writes `eda/`.
- `src/fill_report_numbers.py` reads `split_stats.csv` and `eda/class_totals.csv` and writes the numbers into `report/sections/`.
- `src/train.py` writes `checkpoints/`. `src/evaluate.py` and `src/noise_test.py` read a checkpoint and write `figures/`, `metrics.csv`, `confusion_matrix.*` and `noise_results.*` (Submission 2).
