# report/sections/

One `.tex` file per CRISP-DM section.

Start with `01_business_understanding.tex`. Read in this order:

1. `01_business_understanding.tex`: report section 1
2. `02_data_understanding.tex`: report section 2
3. `03_data_preparation.tex`: report section 3

How they connect:

- `../report.tex` pulls each file in with `\input`, in this order.
- Section 02 includes the `eda_*.png` figures from `../figures/`. Section 03 includes `aug_amplitude_scale.png`.
- The numbers in sections 02 and 03 come from the tables in `results/`.
- Citations use keys from `../references.bib`.
