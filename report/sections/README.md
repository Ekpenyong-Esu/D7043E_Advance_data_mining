# report/sections/

One `.tex` file per CRISP-DM section.

Start with `01_business_understanding.tex`. Read in this order:

1. `01_business_understanding.tex`: report section 1 (Submission 1)
2. `02_data_understanding.tex`: report section 2 (Submission 1)
3. `03_data_preparation.tex`: report section 3 (Submission 1, with the Submission 2 revisions of the validation split and the beat window marked as such)
4. `04_modeling.tex`: baseline, edge model, training configuration, model selection, quantisation
5. `05_evaluation.tex`: DS2 results, per-class scores, quantisation analysis, noise robustness, error analysis, success-criteria review
6. `06_deployment.tex`: MAX78002 compatibility, toolchain, evidence, hardware limitations, proposed validation
7. `07_discussion.tex`: final discussion
8. `08_limitations.tex`: limitations

How they connect:

- `../report.tex` pulls each file in with `\input`, in this order.
- Sections 02 and 03 include the `eda_*.png` and `prep_*.png` figures from `../figures/` and the `eda_*.tex` tables from `../tables/`, both written by `notebooks/EDA.ipynb`.
- Sections 04 to 06 include the other figures and tables in `../figures/` and `../tables/`, written by `notebooks/MODELING.ipynb`. Every number quoted in their text comes from `results/`.
- Citations use keys from `../references.bib`.
