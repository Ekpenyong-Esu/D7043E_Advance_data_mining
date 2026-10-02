# results/

Files written by the pipeline. The report quotes its numbers from here.

Start with `split_stats.csv`. Read in this order:

1. `split_stats.csv`: beats per class per split (report table 3.8).
2. `record_stats.csv`: the same per record, plus edge-dropped and excluded counts.
3. `label_mapping.json`: the symbol to class mapping used.
4. `eda/`: tables for report section 2.

How they connect:

- `make prepare` (`src/prepare_data.py`) writes the first three files at the end of `src/ecg/data/pipeline.py`.
- `make eda` (`src/make_eda_figures.py`) writes `eda/`.
- The report sections in `report/sections/` quote `split_stats.csv` and `eda/class_totals.csv`.
