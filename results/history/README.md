# results/history/

One CSV per CNN run with the training loss and the validation scores (accuracy, macro-F1, per-class precision, recall and F1) after every epoch.

Start with the file of the final run, `<final run>_fp32.csv`. `<final run>_qat.csv` is the QAT fine-tuning.

How they connect: `src/ecg/training/runner.py` returns the history and `src/ecg/training/sweep.py` saves it here. `notebooks/MODELING.ipynb` draws `results/figures/learning_curve.png` from the final run's file. The best epoch of each file is the row in `../experiments.csv`.
