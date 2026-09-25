# report/sections/

One `.tex` file per CRISP-DM section.

Start with `01_business_understanding.tex`. Read in this order:

1. `01_business_understanding.tex`: Submission 1
2. `02_data_understanding.tex`: Submission 1
3. `03_data_preparation.tex`: Submission 1
4. `04_modeling.tex` to `08_limitations.tex`: Submission 2

How they connect:

- `../report.tex` pulls each file in with `\input`, in this order.
- Section 02 includes the `eda_*.png` figures from `../figures/`. Section 03 includes `aug_amplitude_scale.png`.
- `src/fill_report_numbers.py` replaces the `\tbd` placeholders in sections 02 and 03 with numbers from `results/`.
- Citations use keys from `../references.bib`.
