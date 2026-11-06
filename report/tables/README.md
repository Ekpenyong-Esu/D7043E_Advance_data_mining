# results/tables/

LaTeX tables of report sections 4 to 6, written by `notebooks/MODELING.ipynb` with `src/ecg/eda/latex.py` and copied to `report/tables/`.

Start with `model_selection.tex`. In report order:

1. `model_selection.tex`: every sweep run on validation (section 4.4).
2. `model_layers.tex`: the layers of the final CNN (section 4.2).
3. `training_config.tex`: the training settings (section 4.3).
4. `quantization_val.tex`: FP32, PTQ and QAT on validation (section 4.5).
5. `clean_results.tex`, `per_class.tex`, `quantization_ds2.tex`: DS2 results (sections 5.1 to 5.3).
6. `noise_results.tex`, `noise_by_type.tex`: the noise experiment (section 5.4).
7. `error_counts.tex`: the main error types (section 5.5).
8. `success_criteria.tex`: the review (section 5.6).
9. `deployment_evidence.tex`: the toolchain PASS/FAIL table (section 6.3).

The report includes each one with `\input{tables/<name>}`.
