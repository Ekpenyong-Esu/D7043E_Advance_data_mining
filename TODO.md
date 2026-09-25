# TODO — build the project from start to finish

Tick items in order. Each block names the files to touch and the report
section it feeds. Detailed instructions are in [STEP_BY_STEP.md](STEP_BY_STEP.md).

---

## 0. Setup

- [ ] Form the group in Canvas (max 5, week 1). AI-programme rule: the CNN is mandatory.
- [x] `uv venv .venv && uv pip install -r requirements.txt` (system `python -m venv` lacks ensurepip on this machine)
- [x] `make data` → downloads `mitdb` and `nstdb` into `data/raw/`
- [x] `make test` runs (labels and splits tests pass)
- [ ] Fill in the group table in `PROJECT_README.md`

## 1. Business understanding (report §1)

- [x] §1.1 application scenario
- [x] §1.2 input / output / task type
- [x] §1.3 objectives
- [x] §1.4 **measurable** success criteria (macro-F1, S/F recall, noise drop, params, synthesis PASS)
- [x] §1.5 constraints
- [x] §1.6 at least three risks with mitigations

## 2. Data understanding (report §2, `notebooks/EDA.ipynb`)

- [x] §2.1 dataset description table
- [x] §2.2 class mapping table with beat counts (whole DB, DS1, DS2) and documented exclusions (`/`, `f`, `Q`, non-beat symbols)
- [x] §2.3 plots: raw ECG, one beat per class, class distribution, beats per record, amplitude histogram, noisy segments
- [x] §2.4 data quality: NaNs, extreme amplitudes, `~` noise annotations, channel differences, records 102/104 (no MLII), record 114 (channels swapped)
- [x] §2.5 leakage analysis: why beat-level random splits inflate accuracy, how record-level DS1/DS2 prevents it

## 3. Data preparation (report §3, `src/ecg/data/`)

- [x] `filters.py` — median-filter baseline removal, optional 40 Hz low-pass
- [x] `beats.py` — fixed window (90 before, 166 after = 256), drop edge beats
- [x] `normalize.py` — per-beat z-score or max-abs, then clip to the chip input range (about −1 to 1). Wrong input scaling is the most common cause of a large INT8 drop.
- [x] `rr.py` — pre-RR, post-RR and pre-RR / local-mean-RR per beat, from the annotation positions
  - **Why:** S beats have the same shape as N beats; what distinguishes them is that they arrive early. A single-beat window cannot see timing, so RR features are the biggest known lever on S recall in inter-patient evaluation.
  - Use them in the baseline first (cheap). For the CNN, prefer a wider window that covers the neighbouring beats so the timing is visible in the signal itself and the MAX78002 input stays one tensor.
- [x] `make prepare` runs end to end and writes `data/processed/*.npz` + `results/split_stats.csv`
- [x] Un-skip and pass `tests/test_beats.py`
- [x] Check `split_stats.csv`: no record in two splits, all four classes in every split
- [x] §3.1–3.9 written; §3.8 table copied from `split_stats.csv`

## 4. Data augmentation (training records only, `src/ecg/datasets/`)

Applied on the fly inside the train `DataLoader`. Validation and test get **none**.

- [ ] `augment.py`
  - [ ] amplitude scaling ×[0.8, 1.2] — always on, safe
  - [ ] time shift ±5 samples — tolerance to R-peak alignment
  - [ ] baseline wander (low-frequency sinusoid) — prepares for the noise test
  - [ ] light Gaussian noise (σ ≈ 0.02)
  - [ ] optional: real `nstdb` noise at SNR ≥ 6 dB — best for robustness
  - [ ] **do not** time-stretch or warp beats (QRS width is diagnostic)
- [ ] `build_transform()` composes only what `configs/*.yaml` enables
- [ ] `samplers.py` — minority oversampling of S and F
- [ ] `loaders.py` — train loader gets transform + sampler; val/test loaders get neither
- [ ] Plan two training runs: with and without `nstdb` noise augmentation, to answer "could robustness be improved" in §5.4

## 5. Toolchain smoke test (do this before tuning accuracy)

- [ ] Clone and install `ai8x-training` and `ai8x-synthesis` (Linux/WSL)
- [ ] Add dataset loader and `ai85netecg` model to `ai8x-training`
- [ ] 1-epoch train → `quantize.py` → `ai8xize.py` all succeed
- [ ] Copy the working network YAML into `max78002/model_configuration/`

## 6. Submission 1

- [ ] Report §1–3 as PDF, `EDA.ipynb`, `prepare_data.py` + `ecg/data/`, `split_stats.csv`, `PROJECT_README.md`

## 7. Baseline (report §4.1)

- [ ] `training/runner.py` — sklearn branch, input = flattened beat window + RR features from `rr.py`
- [ ] `make baseline` → validation metrics logged to `results/experiments.csv`

## 8. CNN (report §4.2–4.4)

- [ ] **Architecture rule:** only Conv1d, max/avg pooling, ReLU and a small Linear head, fixed 256-sample input, well under the 2 MB weight limit. No Transformers, LSTMs/GRUs, large ResNets or custom ops (GELU, LayerNorm, attention). BatchNorm is fine in training (folded by the toolchain). Practical test: if it does not pass `ai8xize.py`, it is not allowed.
- [ ] `models/ai8x_cnn.py` — **primary definition**, built with ai8x wrappers from day one (they clamp activations to chip ranges even in FP32)
- [ ] `models/cnn.py` — plain-PyTorch mirror with identical Conv1d / pool / ReLU / Linear layers, for use when ai8x is not installed
- [ ] `training/losses.py` — class weights, optional focal loss
- [ ] `training/loop.py` — one epoch train / predict
- [ ] `training/runner.py` — torch branch with early stopping on validation macro-F1
- [ ] `make train` runs; best checkpoint saved
- [ ] Model-selection sweep, all rows in `results/experiments.csv`: window size, model size (channels / kernels / depth), loss function, class balancing (weights / oversampling / focal), normalization method, filtering on/off, augmentation on/off
- [ ] Keep one small and one larger CNN checkpoint from the sweep (both synthesisable) for the robustness-vs-complexity comparison in block 11
- [ ] If validation S recall misses the success criterion: switch to a multi-beat window (previous + current + next beat) before any other change. See `APPROACHES.md` §4.2 B.
- [ ] Re-run the Step 5 smoke test after every architecture change
- [ ] **Freeze the model.** Final architecture, hyperparameters and checkpoint are chosen by validation macro-F1 on DS1 only. No DS2 result has been seen at this point.
- [ ] §4.2 layer table, §4.3 training config, §4.4 selection written

## 9. Quantization (report §4.5, `max78002/`)

- [ ] Train the chosen architecture inside `ai8x-training` (FP32)
- [ ] PTQ: `quantize.py --clip-method MSE` → evaluate with `-8` on the **validation** set → record INT8 macro-F1
- [ ] If the drop is too large: QAT as a **fine-tune from the FP32 checkpoint** for the last 10–20 epochs (`--qat-policy`, `qat_start_epoch`), not from scratch; quantize, evaluate again on validation
- [ ] **Freeze the INT8 model.** The FP32-vs-INT8 decision is made on validation, never on DS2.
- [ ] Logs and `-q.pth.tar` files in `max78002/quantization_output/`
- [ ] §4.5 written: FP32 vs PTQ-INT8 (vs QAT-INT8), why QAT was or was not needed

## 10. Final evaluation on DS2 (report §5.1–5.3)

- [ ] `evaluation/confusion.py`, `report.py`, `runner.py`
- [ ] `make evaluate` (uses `--final`) **once** per frozen model: baseline, FP32, INT8
- [ ] **Do not retrain or re-tune after seeing DS2 numbers.** A DS2 score below validation is the expected inter-patient gap; report it as it is and discuss it in §5.2 / §7.
- [ ] `results/metrics.csv` and `results/confusion_matrix.*` produced
- [ ] §5.1, §5.2, §5.3 tables filled; discussion of hard classes (S, F)

## 11. Noise robustness (report §5.4)

- [ ] `noise/snr.py` — scale noise to target SNR on the **raw** signal, fixed seed
- [ ] Un-skip and pass `tests/test_snr.py`
- [ ] `noise/experiment.py` — raw DS2 + noise → same preprocessing → frozen model
- [ ] `make noise` → `results/noise_results.csv` and `.png` (clean / 12 / 6 / 0 dB, per noise type)
- [ ] Run once with filtering on, once with filtering off
- [ ] Run for the noise-augmented model from block 4
- [ ] Robustness vs model complexity: run the noise test on the small and the larger CNN from block 8; report macro-F1 at each SNR against parameter count
- [ ] §5.4 written: which classes degrade first, gradual or not, does preprocessing help, does augmentation help

## 12. Error analysis and criteria review (report §5.5–5.6)

- [ ] `evaluation/errors.py` — pick N→S, S→N, V→x, and 0 dB errors; plot waveforms
- [ ] §5.5 written with figures
- [ ] §5.6 each success criterion marked Achieved / Partially / Not achieved with justification

## 13. Deployment (report §6)

- [ ] Final synthesis: `ai8xize.py` with `--sample-input` (`deployment/sample_input.py`)
- [ ] `max78002/synthesis_output/` holds C files, `weights.h`, log
- [ ] `deployment/evidence.py` → §6.3 PASS/FAIL table with weight and data memory
- [ ] §6.1 compatibility, §6.4 software-verified vs hardware-verified (no latency / power claims), §6.5 proposed hardware validation

## 14. Discussion, limitations, package

- [ ] §7 discussion, §8 limitations (every bullet in the assignment answered)
- [ ] `report.pdf` exported to the project root
- [ ] Final checklist: no `data/` in the zip, `make prepare` works from a clean checkout, every number traceable to `results/` or `max78002/`, no clinical or hardware claims
- [ ] Zip as `GroupXX_ECG_Project/` and submit
