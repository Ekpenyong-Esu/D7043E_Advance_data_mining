# report/

LaTeX sources for the report. Group 16, D7043E, ECG Heartbeat Classification for Edge AI on the MAX78002.

Start with `report.tex`. Then read in this order:

1. `report.tex`: main file and preamble. Compiles to `report.pdf` in the project root.
2. `sections/`: one file per CRISP-DM section. See `sections/README.md`.
3. `references.bib`: the bibliography.
4. `thesisMDH.cls`: document class from the LTU/MDH template.
5. `figures/` and `images/`: pictures the sections and the template use.
6. `tables/`: LaTeX tables written and copied in by the notebooks.

```text
report/
├── report.tex
├── thesisMDH.cls
├── references.bib
├── sections/
│   ├── 01_business_understanding.tex
│   ├── 02_data_understanding.tex
│   ├── 03_data_preparation.tex
│   ├── 04_modeling.tex
│   ├── 05_evaluation.tex
│   ├── 06_deployment.tex
│   ├── 07_discussion.tex
│   └── 08_limitations.tex
├── figures/
├── tables/
└── images/
```

How they connect:

- `report.tex` loads `thesisMDH.cls`, then pulls in each file in `sections/` with `\input`, then the bibliography from `references.bib`.
- The sections include figures from `figures/` and tables from `tables/`. `notebooks/EDA.ipynb` writes the ones for sections 2 and 3, `notebooks/MODELING.ipynb` the ones for sections 4 to 6.
- The numbers in the text come from `results/` and `max78002/`.
- `thesisMDH.cls` draws the title page with the logos in `images/`.

`thesisMDH.cls` and `images/` come from the template. The only change is that the `swedish` babel option was removed. It was unused and needs a language pack that is not installed by default.

The last section of each notebook builds the PDF. To rebuild only the PDF, use the LaTeX Workshop button in VS Code (recipe in `.vscode/settings.json`), or from the project root:

```bash
make report
```
