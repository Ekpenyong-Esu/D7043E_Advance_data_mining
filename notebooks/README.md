# notebooks/

Start with `EDA.ipynb`. It is the only notebook.

- `EDA.ipynb`: exploratory data analysis for report section 2.

How it connects: the notebook imports `stats` and `plots` from `src/ecg/eda/` and calls them on the raw records in `data/raw/mitdb/`. It defines no logic of its own. It is the only code that draws figures. It writes CSV tables to `results/eda/`, figures to `results/eda/figures/` and LaTeX tables to `results/eda/tables/`; copy them into `report/` by hand. Run the notebook with the `.venv` kernel.
