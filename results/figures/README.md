# results/figures/

Figures of report sections 4 to 6, drawn by `notebooks/MODELING.ipynb` with `src/ecg/evaluation/plots.py`. Section 10 of the notebook copies them to `report/figures/`.

Start with `learning_curve.png`. In report order:

1. `learning_curve.png`: training loss and validation scores per epoch of the final run (section 4.3).
2. `fp32_vs_int8.png`: macro-F1 and per-class recall of the FP32 and INT8 models on DS2 (section 5.3).
3. `confusion_matrix_<model>.png`: one per model on DS2 (section 5.2).
4. `noise_results.png`: macro-F1 against SNR per noise type, and per-class F1 (section 5.4).
5. `error_examples.png` and `error_examples_noise.png`: misclassified beats, clean and at 0 dB (section 5.5).
