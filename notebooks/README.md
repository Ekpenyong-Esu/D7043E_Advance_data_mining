# notebooks/

Start with `EDA.ipynb`. It is the only notebook.

- `EDA.ipynb`: exploratory data analysis for report section 2.

How it connects: the notebook imports `stats` and `plots` from `src/ecg/eda/` and calls them on the raw records in `data/raw/mitdb/`. It defines no logic of its own. `make eda` runs the same functions from `src/make_eda_figures.py` and writes the figures to `report/figures/` and the tables to `results/eda/`. Run the notebook with the `.venv` kernel.
