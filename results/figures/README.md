# results/figures/

Plots for Submission 2: confusion matrix, noise curve, error examples.

How they connect: `src/evaluate.py` writes the confusion matrix and error examples through `src/ecg/evaluation/`. `src/noise_test.py` writes the noise curve through `src/ecg/noise/experiment.py`. The ones used in the report are copied to `report/figures/`.
