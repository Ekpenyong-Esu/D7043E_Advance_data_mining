# Step-by-Step Guide: ECG Heartbeat Classification on MAX78002

This document walks through the whole project in execution order, mapped to
the six CRISP-DM stages and the two course submissions.

```
Submission 1  ──►  Steps 1–6   (Business Understanding, Data Understanding, Data Preparation)
Submission 2  ──►  Steps 7–13  (Modeling, Evaluation, Deployment, Discussion)
```

CRISP-DM is iterative: if Step 8 or 10 reveals a problem, go back to Step 5
or 6, change the preprocessing, and document what changed and why.

---

## Step 0 — Set up the group and the repository

1. Form a group of at most 5 people in Canvas (People section) during week 1.
   PhD students work alone. Confirm nobody has done this project before.
   **AI-programme students (TCAIA, TMDIA) must use deep learning.** The
   classical baseline in Step 7 is only a reference; the 1D CNN in Step 8 is
   the mandatory deliverable, so do not treat it as optional.
2. Create the project folder with the required layout:

   ```bash
   mkdir -p GroupXX_ECG_Project/{notebooks,src,configs,results,max78002,data/raw,data/processed}
   cd GroupXX_ECG_Project
   git init
   printf "data/\n.venv/\n*.pth\n__pycache__/\n" > .gitignore
   ```

3. Create a virtual environment and `requirements.txt`:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install wfdb numpy scipy pandas scikit-learn matplotlib seaborn torch torchvision pyyaml jupyter tqdm
   pip freeze > requirements.txt
   ```

4. Decide on a single random seed (e.g. `42`) and put it in `configs/preprocess.yaml`.
   Every script must read the seed from the config, never hard-code it.

---

## Step 1 — Business / Application Understanding (Report §1)

Write these sections before touching data. They frame every later decision.

| Section | What to write |
|---------|---------------|
| 1.1 Application scenario | Wearable cardiac monitoring; why on-device inference (privacy, latency, battery, no connectivity); who would use it (patients, clinicians, device makers, researchers). |
| 1.2 Data-mining objective | **Input:** one fixed-length ECG segment (e.g. 256 samples at 360 Hz, single lead MLII) centred on an annotated R-peak. **Output:** one of {N, S, V, F}. **Task:** multiclass classification. |
| 1.3 Project objectives | Distinguish N/S/V/F; generalize to unseen records; robust to noise; compact model; MAX78002-compatible. |
| 1.4 Success criteria | Must be **measurable**. Example: macro-F1 ≥ 0.70 on DS2; S and F recall ≥ 0.50; macro-F1 drop ≤ 0.15 at 6 dB SNR; ≤ 100 k parameters and ≤ 2 MB; `ai8xize.py` completes with PASS. |
| 1.5 Constraints | Edge memory, INT8 arithmetic, class imbalance, 47 subjects only, noise, compute budget, no physical board. |
| 1.6 Risks | Minimum three, in a table: risk / why it matters / mitigation. Good candidates: severe S/F imbalance; inter-patient domain shift; quantization accuracy loss; synthesis failure; noise sensitivity. |

**Tip:** write the success criteria as a table now. Step 12 will reuse it verbatim.

---

## Step 2 — Download the data

```bash
python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"
```

Sanity check:

```python
import wfdb
rec = wfdb.rdrecord('data/raw/mitdb/100')
ann = wfdb.rdann('data/raw/mitdb/100', 'atr')
print(rec.fs, rec.sig_name, rec.p_signal.shape, len(ann.sample))
# 360 ['MLII', 'V5'] (650000, 2) 2274
```

From `nstdb` you only need the three pure-noise records: `bw` (baseline
wander), `em` (electrode motion), `ma` (muscle artifact). Ignore the pre-mixed
`118e*` / `119e*` records; they are DS1 records and would leak.

---

## Step 3 — Data Understanding: EDA (Report §2, `notebooks/EDA.ipynb`)

Produce every figure and table below; they are all explicitly required.

1. **Dataset description table** (§2.1): 48 records, 47 subjects, 2 channels,
   360 Hz, ~30 min, beat annotations, leads (MLII in 46 records; V1/V2/V4/V5 as
   second lead; records 102 and 104 have no MLII).
2. **Class mapping table** (§2.2). Standard AAMI mapping:

   | Final class | MIT-BIH symbols |
   |-------------|-----------------|
   | N | `N`, `L`, `R`, `e`, `j` |
   | S | `A`, `a`, `J`, `S` |
   | V | `V`, `E` |
   | F | `F` |
   | *excluded* | `/`, `f`, `Q`, and non-beat annotations (`+`, `~`, `|`, `"`, `x`, `[`, `]`, `!` …) |

   Count beats per class for the whole database, DS1 and DS2.
3. **Plots** (§2.3):
   - 10 s of raw ECG from one record, both channels.
   - One example beat per class (N, S, V, F), aligned at the R-peak.
   - Class distribution bar chart (log scale; F is tiny).
   - Beats per record (stacked bar by class) → shows how few records contain S/F.
   - Amplitude histogram (per record and overall).
   - Segments from records known to be noisy (e.g. 104, 203, 207, 208).
4. **Data quality** (§2.4): check for NaNs, flat lines, clipping / extreme
   amplitudes, `~` noise annotations, differences between channels, records
   without MLII (102, 104 — both paced, so already excluded by class mapping).
5. **Leakage analysis** (§2.5): explain that beats from the same person share
   morphology, so a random beat split lets the model memorize subjects and
   reports inflated accuracy (often >99 %) that collapses inter-patient
   (typically 80–90 %). State that you split **by record**, using DS1/DS2,
   and that validation inside DS1 is also by record.

---

## Step 4 — Data Preparation: `src/prepare_data.py` (Report §3)

Implement one script that turns raw records into fixed-size arrays.

### 4.1 Record selection (§3.1)

```yaml
# configs/preprocess.yaml
seed: 42
fs: 360
lead: MLII
ds1: [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
ds2: [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
val_records: [108,116,124,203,208,209,223]   # subset of DS1, chosen so S and V appear in validation
```

Train = DS1 minus `val_records`. Validation = `val_records`. Test = DS2.
Make sure validation contains S and V beats (S is concentrated in a few
records such as 118, 124, 207, 209, 232).

### 4.2 Channel selection (§3.2)

Use **MLII**. Select it **by name** from `record.sig_name`, never by index:
in record **114** the channels are swapped (V5 first, MLII second). Justify:
available in 46/48 records, best QRS visibility. If MLII is missing (102, 104), use the first
channel *or* exclude the record; since both are paced they are excluded anyway.
Document this.

### 4.3 Beat extraction (§3.3)

R-peak locations come from the `.atr` annotation `sample` field (expert
annotations, no detector needed). For each beat:

```
window = signal[r - before : r + after]      # e.g. before = 90, after = 166 → 256 samples (~0.71 s)
```

Discard beats too close to the record edges. Keep the window a power of two
or a multiple of 4 — it simplifies the MAX78002 layout.

### 4.4 Signal preprocessing order (§3.4)

```
Raw ECG (360 Hz, MLII)
 → 1. baseline-wander removal  (high-pass 0.5 Hz or median-filter 200 ms + 600 ms)   # remove drift
 → 2. optional low-pass 40 Hz                                                        # remove EMG / mains
 → 3. segmentation around R-peak (fixed window)                                     # fixed-size input
 → 4. per-beat normalization (z-score or divide by max|x|)                          # amplitude varies per patient / electrode
 → 5. clipping to [-1, 1]  (or [-4, 4] σ)                                          # bounded range for INT8 (ai8x expects [-128, 127] / 128)
 → model input, shape (1, 256)
```

Write one sentence of justification per operation in the report.

### 4.5 Label preparation (§3.5)

Implement the mapping from §2.2 as a dictionary in code and dump it to
`results/label_mapping.json` so it is reproducible.

### 4.6 Class imbalance (§3.6)

Typical DS1 counts: N ≈ 45 800, S ≈ 940, V ≈ 3 800, F ≈ 410. Report the ratio.
Recommended strategy: **class-weighted cross-entropy** (weights ∝ 1/√freq or
inverse freq) *plus* mild oversampling of S and F in the training sampler.
Alternative: focal loss. Justify the choice; you can compare in Step 8.

### 4.7 Augmentation (§3.7) — training set only

- Amplitude scaling ×[0.8, 1.2]
- Temporal shift ±5 samples
- Additive Gaussian noise (σ ≈ 0.02)
- Baseline wander: low-frequency sinusoid
- Optionally add `nstdb` noise at random SNR ≥ 6 dB (this is allowed for
  training as long as the test records are never involved).

Implement in `src/dataset.py` and switch off for validation / test.

### 4.8 Output

`prepare_data.py` writes:

```
data/processed/train.npz   # X: (n, 1, 256) float32, y: (n,) int64, record_id: (n,)
data/processed/val.npz
data/processed/test.npz
results/split_stats.csv    # Split | records | N | S | V | F | Total   (→ report §3.8)
results/label_mapping.json
```

### 4.9 Reproducibility (§3.9)

Record in the report: Python and package versions (`pip freeze`), seed,
filter parameters, window (`before`/`after`), normalization method, fs,
input size.

---

## Step 5 — Verify the splits before Submission 1

```bash
python src/prepare_data.py --config configs/preprocess.yaml
cat results/split_stats.csv
```

Check:
- No record appears in more than one split.
- Every split has all four classes (F may be small in validation; note it).
- Total beat count ≈ 100 k after exclusions.

---

## Step 5b — Toolchain smoke test (do this before optimizing accuracy)

Synthesis failure is the biggest project risk, so prove the pipeline
end-to-end first. Install the toolchain (Step 9.1), register the dataset and
the Step 8.1 model (Step 9.2), then:

```bash
python train.py --epochs 1 --model ai85netecg --dataset ECG_MITBIH --device MAX78002 ...
python quantize.py <ckpt> <out> --device MAX78002
python ai8xize.py --config-file networks/ecg-max78002.yaml --checkpoint-file <out> --device MAX78002 ...
```

If any step fails, fix the architecture now. From here on, re-run this check
after every architecture change. See [APPROACHES.md](APPROACHES.md) §2.1.

---

## Step 6 — Submission 1 package

Deliver: report §1–3 (PDF), `notebooks/EDA.ipynb`, `src/prepare_data.py`,
`src/dataset.py`, `results/split_stats.csv`, `README.md`. No model results
required yet.

---

## Step 7 — Baseline (Report §4.1)

Train something simple on the same processed arrays so the CNN has a
reference point:

- **Logistic regression** or **random forest** on the raw 256-sample window
  (optionally plus a few hand-crafted features: pre/post RR interval, QRS width).
- Evaluate on **validation** for now; DS2 only at the end.

Save the model and its validation metrics; `metrics.csv` will get a Baseline row later.

---

## Step 8 — Compact 1D-CNN (Report §4.2–4.4)

### 8.1 Architecture (`src/model.py`)

A starting point that fits the accelerator and stays well under 100 k parameters:

| Layer | Input | Operation | Output | Params |
|-------|-------|-----------|--------|--------|
| 1 | 1×256 | Conv1d k=7, 16 ch, ReLU | 16×256 | 128 |
| 2 | 16×256 | MaxPool 2 | 16×128 | 0 |
| 3 | 16×128 | Conv1d k=5, 32 ch, ReLU | 32×128 | 2 592 |
| 4 | 32×128 | MaxPool 2 | 32×64 | 0 |
| 5 | 32×64 | Conv1d k=3, 64 ch, ReLU | 64×64 | 6 208 |
| 6 | 64×64 | MaxPool 4 | 64×16 | 0 |
| 7 | 64×16 | Conv1d k=3, 64 ch, ReLU | 64×16 | 12 352 |
| 8 | 64×16 | AvgPool 16 (or flatten) | 64×1 | 0 |
| 9 | 64 | Linear → 4 | 4 | 260 |

Use **only** Conv1d, pooling, ReLU, Linear; no BatchNorm at inference
(ai8x fuses BN into conv, so BN is fine during training if you use the ai8x
wrappers), no residuals unless you verify support, no LSTM / attention.

Write the model **once**, using `ai8x.FusedConv1dReLU`,
`ai8x.FusedMaxPoolConv1dReLU`, `ai8x.Linear` from the `ai8x-training` repo,
and import that same class from `src/model.py`. With QAT off the wrappers
behave like ordinary PyTorch layers, so there is no need for a second copy.
(See [APPROACHES.md](APPROACHES.md) §2.3 for the trade-offs.)

### 8.2 Training configuration (`src/train.py`, `configs/cnn_fp32.yaml`)

```yaml
optimizer: Adam
lr: 1e-3
scheduler: cosine or ReduceLROnPlateau
batch_size: 128
epochs: 50
loss: weighted cross-entropy
early_stopping: patience 10 on validation macro-F1
augmentation: as in Step 4.7
seed: 42
```

### 8.3 Model selection (§4.4)

Vary and log (use a small table in the report):
window length (128 / 256 / 360), number of channels, kernel size, loss
(weighted CE vs focal), oversampling on/off, filtering on/off.
**Selection metric: validation macro-F1.** Never look at DS2 here.

Save the best checkpoint to `results/checkpoints/cnn_fp32_best.pth`.

---

## Step 9 — MAX78002 toolchain: QAT, quantization, synthesis (Report §4.5, §6)

### 9.1 Install the toolchain (Linux / WSL recommended)

```bash
git clone --recursive https://github.com/analogdevicesinc/ai8x-training.git
git clone --recursive https://github.com/analogdevicesinc/ai8x-synthesis.git

cd ai8x-training && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # requirements-cpu.txt if no GPU
cd ../ai8x-synthesis && python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Follow the README in each repo; versions of PyTorch are pinned and matter.

### 9.2 Add your dataset and model to ai8x-training

1. `datasets/ecg_mitbih.py` — loader that reads `data/processed/*.npz`,
   returns tensors shaped `(1, 256)` scaled to the ai8x input range, and
   registers itself in the `datasets` list with `input: (1, 256)`, `output: (0,1,2,3)`.
2. `models/ai85net-ecg.py` — the architecture from Step 8 using `ai8x.*`
   layers, registered in the `models` list with `min_input: 1`, `dim: 1`.
3. Policy files: copy `policies/schedule.yaml` and a `qat_policy` (e.g. start
   QAT at epoch 30, `weight_bits: 8`).

### 9.3 Post-training quantization first, then QAT only if needed

Quantize the **FP32** checkpoint directly and measure the drop before
committing to QAT:

```bash
cd ai8x-synthesis && source venv/bin/activate
python quantize.py ../ai8x-training/logs/<fp32 run>/best.pth.tar \
  trained/ai85-ecg-ptq8-q.pth.tar --device MAX78002 --clip-method MSE -v
```

Evaluate it with the command in 9.6. If macro-F1 or minority-class recall
drops more than your success criterion allows, proceed to QAT below and
report all three: FP32, PTQ-INT8, QAT-INT8. This gives the evidence the
assignment asks for in "explain why QAT is needed".

### 9.4 Train with QAT

```bash
python train.py --lr 0.001 --optimizer adam --epochs 50 --deterministic \
  --compress policies/schedule-ecg.yaml --qat-policy policies/qat_policy_ecg.yaml \
  --model ai85netecg --dataset ECG_MITBIH --device MAX78002 --batch-size 128 \
  --confusion --pr-curves --enable-tensorboard
```

Explain in the report **why QAT is used**: post-training INT8 quantization of
a small network can lose accuracy, especially for minority classes; QAT
simulates 8-bit weights during training so the network adapts.

### 9.5 Quantize the QAT checkpoint (ai8x-synthesis)

```bash
cd ai8x-synthesis && source venv/bin/activate
python quantize.py ../ai8x-training/logs/<run>/qat_best.pth.tar \
  trained/ai85-ecg-qat8-q.pth.tar --device MAX78002 -v
```

Store the log and the `-q.pth.tar` in `max78002/quantization_output/`.

### 9.6 Evaluate the INT8 checkpoint

```bash
cd ai8x-training && source venv/bin/activate
python train.py --model ai85netecg --dataset ECG_MITBIH --device MAX78002 \
  --evaluate --exp-load-weights-from ../ai8x-synthesis/trained/ai85-ecg-qat8-q.pth.tar \
  -8 --confusion
```

This gives the **INT8 macro-F1** for report §5.3.

### 9.7 Write the network YAML and synthesize

`max78002/model_configuration/ecg-max78002.yaml` describes every layer
(processors, output offsets, operations). Then:

```bash
python ai8xize.py --verbose --test-dir synthesized --prefix ecg-max78002 \
  --checkpoint-file trained/ai85-ecg-qat8-q.pth.tar \
  --config-file networks/ecg-max78002.yaml \
  --sample-input tests/sample_ecg_mitbih.npy \
  --device MAX78002 --compact-data --mexpress --softmax --overwrite
```

`--sample-input` needs one preprocessed beat saved as `.npy` (int8-scaled,
shape `(1, 256)`); generate it with a small script and keep it in the repo.

Copy `synthesized/ecg-max78002/` (C files, `weights.h`, `cnn.c`, `main.c`,
log) into `max78002/synthesis_output/`.

### 9.8 Fill the deployment evidence table (§6.3)

| Requirement | Result | Evidence |
|-------------|--------|----------|
| Model accepted by toolchain | PASS / FAIL | training log |
| Quantization completed | PASS / FAIL | `quantize.py` log |
| Synthesis completed | PASS / FAIL | `ai8xize.py` log |
| C / device files generated | PASS / FAIL | `ls synthesis_output/` |
| Weight memory requirement | ___ bytes | `ai8xize.py` verbose output |
| Data memory requirement | ___ bytes | `ai8xize.py` verbose output |

If synthesis fails, read the error, fix the layer (common causes: unsupported
kernel size, output too large for a processor group, pooling before the first
conv, odd input length), retrain, repeat. Document each failure.

---

## Step 10 — Final evaluation on DS2 (Report §5.1–5.3, `src/evaluate.py`)

Only now touch DS2. Run once per model; no tuning afterwards.

```bash
python src/evaluate.py --checkpoint results/checkpoints/baseline.pkl      --split test --name baseline
python src/evaluate.py --checkpoint results/checkpoints/cnn_fp32_best.pth --split test --name cnn_fp32
python src/evaluate.py --checkpoint <int8 checkpoint or exported logits>   --split test --name cnn_int8
```

`evaluate.py` must output, per model: accuracy, macro-F1, per-class
precision / recall / F1, confusion matrix (`results/confusion_matrix.png`
and `.csv`), parameter count → all rows appended to `results/metrics.csv`.

Report:
- §5.1 table: Baseline vs FP32 vs INT8 (accuracy, macro-F1, parameters).
- §5.2 per-class table and discussion (S is usually hardest: morphologically
  close to N, few training subjects; F is rarest).
- §5.3 FP32 macro-F1, INT8 macro-F1, difference.

---

## Step 11 — Noise robustness (Report §5.4, `src/noise_test.py`)

1. Load `bw`, `em`, `ma` from `nstdb` (360 Hz already; two channels each).
2. For each **DS2 record** (raw signal, *before* preprocessing) and each SNR
   in {12, 6, 0} dB, scale the noise so that

   ```
   SNR_dB = 10 · log10( P_signal / P_noise )
   ```

   using the power of the clean ECG segment, add it, then run the **same**
   preprocessing pipeline and the final model.
3. Use a fixed seed for noise offsets. Never use these noisy signals for
   training the final model.
4. Report the table Clean / 12 dB / 6 dB / 0 dB with accuracy and macro-F1,
   plus per-class F1, and plot macro-F1 vs SNR (`results/noise_results.png`
   and `.csv`).
5. Discuss: which classes degrade first, whether degradation is gradual, whether
   the filtering from Step 4.4 helps (compare with filtering disabled), and
   what would help (noise augmentation in training, larger window, etc.).

---

## Step 12 — Error analysis and success-criteria review (Report §5.5–5.6)

1. From the DS2 confusion matrix, pull indices of: N→S, S→N, V→N or V→F,
   and errors that only appear at 0 dB. Plot 6–8 waveforms with true and
   predicted labels, and comment on the morphology (e.g. S beats with nearly
   normal QRS but short RR; V beats in records with unusual morphology).
2. Copy the success-criteria table from Step 1 and mark each row
   **Achieved / Partially achieved / Not achieved** with a one-line justification.

---

## Step 13 — Deployment write-up and discussion (Report §6–8)

- §6.1 Compatibility: list layer types, input `(1, 256)`, output `4`, parameter count, size in bytes.
- §6.4 Hardware limitations: make a two-column list, **Software verified** (quantized accuracy in Python, synthesis output, memory estimates from the tool) vs **Hardware verified** (nothing — no board). Do not report latency, power, or energy numbers.
- §6.5 Proposed hardware validation: flash the generated project with the MSDK, feed the same `sample_ecg_mitbih.npy` used in `ai8xize.py --sample-input`, compare the on-device output against the known-answer test in the generated `main.c`, then loop over many DS2 beats over serial, and measure time and current with the EV kit's power monitor.
- §7 Discussion and §8 Limitations: answer every bullet in the assignment
  text; use the numbers from `results/`.

---

## Step 14 — Final submission package

```
GroupXX_ECG_Project/
├── report.pdf            ← §1–8 complete, all tables filled
├── README.md             ← data download + reproduction instructions
├── STEP_BY_STEP.md
├── requirements.txt
├── notebooks/EDA.ipynb
├── src/{prepare_data,dataset,model,train,evaluate,noise_test}.py
├── configs/*.yaml
├── results/{metrics.csv, confusion_matrix.*, noise_results.*, split_stats.csv}
└── max78002/{model_configuration, quantization_output, synthesis_output}
```

Before zipping:

- [ ] `data/` is **not** included.
- [ ] `python src/prepare_data.py` runs from a clean checkout after the downloads in the README.
- [ ] Every number in the report can be traced to a file in `results/` or `max78002/`.
- [ ] No sentence claims clinical validation or measured hardware performance.
- [ ] All six CRISP-DM stages appear as headings in the report.

---

## Quick command summary

| Stage | Command |
|-------|---------|
| Download | `python -c "import wfdb; wfdb.dl_database('mitdb','data/raw/mitdb')"` |
| Prepare | `python src/prepare_data.py --config configs/preprocess.yaml` |
| EDA | `jupyter notebook notebooks/EDA.ipynb` |
| Baseline | `python src/train.py --config configs/baseline.yaml` |
| CNN FP32 | `python src/train.py --config configs/cnn_fp32.yaml` |
| QAT (ai8x) | `python train.py ... --qat-policy policies/qat_policy_ecg.yaml --device MAX78002` |
| Quantize | `python quantize.py <qat ckpt> <out> --device MAX78002` |
| Synthesize | `python ai8xize.py --config-file networks/ecg-max78002.yaml --device MAX78002 ...` |
| Test (DS2) | `python src/evaluate.py --checkpoint ... --split test` |
| Noise | `python src/noise_test.py --checkpoint ...` |
