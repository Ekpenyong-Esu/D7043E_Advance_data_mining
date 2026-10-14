# results/eda/

Tables for report section 2, written by `notebooks/EDA.ipynb`.

Start with `class_totals.csv`. Read in this order:

1. `class_totals.csv`: N/S/V/F counts and percentages for DS1, DS2 and all.
2. `class_counts_per_record.csv`: per-record counts and split membership.
3. `symbol_counts.csv`: raw annotation symbol counts before mapping.
4. `leads.csv`: channel names, sampling rate and length per record.
5. `amplitude.csv`: amplitude range and median R-peak height per record.
6. `noise_marks.csv`: number of `~` noise annotations per record.

How they connect:

- `notebooks/EDA.ipynb` calls the functions in `src/ecg/eda/stats.py` on the raw records and saves each returned table here.
- `class_totals.csv` is built from `class_counts_per_record.csv`, which is built with the same mapping as `../label_mapping.json`.
- The figures in `figures/` are drawn from these same tables by `src/ecg/eda/plots.py`.
- The class table in report section 2.2 quotes `class_totals.csv`.
- `notebooks/EDA.ipynb` also writes `figures/*.png` and LaTeX tables `tables/*.tex`
  (made by `src/ecg/eda/latex.py`). Copy them by hand to `report/figures/` and
  `report/tables/`.
