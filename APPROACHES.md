# Approaches: Review of STEP_BY_STEP.md and Alternative Ways to Solve the Project

This file answers three questions about [STEP_BY_STEP.md](STEP_BY_STEP.md):

1. Does it cover the project requirements?
2. Is it the simplest good way to do it?
3. Is it how this kind of work is done in industry?

Then it lists the realistic alternatives for each major decision so the group
can choose deliberately and justify the choice in the report.

---

## 1. Requirement coverage — PASS

Every numbered section of the assignment maps to a step:

| Assignment section | Guide step | Covered |
|--------------------|-----------|---------|
| §1 Business understanding (1.1–1.6) | Step 1 | Yes |
| §2 Data understanding (2.1–2.5) | Steps 2–3 | Yes |
| §3 Data preparation (3.1–3.9) | Steps 4–5 | Yes |
| §4 Modeling (4.1–4.5) | Steps 7–9 | Yes |
| §5 Evaluation (5.1–5.6) | Steps 10–12 | Yes |
| §6 Deployment (6.1–6.5) | Steps 9, 13 | Yes |
| §7–8 Discussion, limitations | Step 13 | Yes |
| §9 Package layout | Step 14 | Yes |
| Hard rules (no beat mixing, DS2 untouched, no accuracy-only, no clinical / hardware claims) | Steps 4.1, 10, 13, 14 | Yes |

Nothing required is missing.

---

## 2. Is it the simplest way? — Mostly, with three corrections

### 2.1 The MAX78002 toolchain is used too late (biggest issue)

The guide only touches `ai8x-training` / `ai8x-synthesis` at Step 9, after
model selection. The assignment says a model that cannot be synthesized
"does not fully satisfy the project objective", so synthesis failure is the
single biggest project risk, and the guide discovers it last.

**Industry practice:** get a trivial version working end-to-end first
("walking skeleton" / "tracer bullet"), then improve accuracy. In embedded ML
this means: tiny model → quantize → synthesize → generated C compiles, on day
one of modeling. Then every later architecture change is checked against the
toolchain immediately.

**Fix:** insert a *toolchain smoke test* right after the data split (Step 5).
Train the Step 8.1 network for 1 epoch inside `ai8x-training`, run
`quantize.py` and `ai8xize.py`. If it fails, fix the architecture before
spending days on accuracy. This is now noted in the guide.

### 2.2 QAT is chosen before it is justified

The guide goes straight to Quantization-Aware Training. The assignment says
"*If* QAT is used, explain *why* QAT is needed", which only makes sense if
you first measure what plain post-training quantization (PTQ) loses.

**Industry practice:** PTQ first (it is free: one command on the FP32
checkpoint). Only if the INT8 drop is unacceptable, especially on minority
classes, switch to QAT. This also produces a better §4.5 / §5.3 discussion:
FP32 → PTQ INT8 → QAT INT8.

**Fix:** Step 9 order should be: train FP32 in ai8x-training → `quantize.py`
with a clipping method (PTQ) → evaluate `-8` → decide on QAT.

### 2.3 Two model implementations is extra work

The guide says to write the model twice (plain PyTorch in `src/model.py` and
again with `ai8x.*` layers). Two copies drift apart and double the debugging.

**Simplest:** write it once with the `ai8x` layer wrappers and import that
same class from `src/model.py` (or vendor the single file `ai8x.py`). The
ai8x wrappers behave as normal PyTorch modules when QAT is off, so fast
iteration is still possible.

### 2.4 Factual trap: channel index

The guide said MLII is "channel 0". In record **114** the channels are swapped
(V5 first, MLII second). Always select the channel by name from
`record.sig_name`, never by index. The guide is corrected.

---

## 3. Is it industry practice? — Yes for the method, partly for the tooling

CRISP-DM is what the course requires, and it is still the dominant process
model in applied data-science teams. The train/validate/test-by-subject
discipline, the fixed-seed configs, the metrics-to-CSV outputs and the
"software verified vs hardware verified" distinction are all standard.

What an industry team would add, none of it required by the course, listed
from cheapest to most effort:

| Practice | Why | Cost |
|----------|-----|------|
| Smoke test the whole pipeline first (see 2.1) | Removes the biggest risk early | 1 day |
| One experiment log (`results/experiments.csv` or MLflow / Weights & Biases) | Model-selection table in §4.4 writes itself | 1 h |
| Config-driven runs, no hard-coded numbers | Reproducibility, §3.9 | already in guide |
| `evaluate.py --split test` refuses to run unless `--final` flag is passed | Enforces "DS2 untouched" | 10 min |
| Unit tests for preprocessing and label mapping (`pytest`) | Catches off-by-one windows, wrong symbol maps | 2 h |
| Version-pin `requirements.txt` and record git commit hash in every results file | Traceability | 30 min |
| Known-answer test of the generated C against Python output (the ai8xize `main.c` already contains one) | The only "hardware-like" check available without a board | included in synthesis |

---

## 4. Alternative ways to solve each part

Pick one per row. The **bold** option is the recommended simple path.

### 4.1 Where the code lives

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. ai8x-first** | Dataset loader + model + training all inside `ai8x-training`; `src/*.py` are thin wrappers or copies | One source of truth; QAT, quantize, synth all native | Heavy pinned environment; Linux/WSL needed |
| B. PyTorch-first, port later | Own `src/train.py`; re-implement in ai8x at the end | Fast iteration; clean course layout | Duplicate model; port surprises late (what the original guide did) |
| C. Hybrid | Own `src/` scripts that `import ai8x` from the cloned repo | Single model definition, own scripts | Must keep ai8x import path working |

### 4.2 Model family

| Option | Description | Expected S/F performance | MAX78002 fit |
|--------|-------------|--------------------------|--------------|
| **A. Single-beat 1D CNN** (guide Step 8) | 256-sample window around R-peak | Moderate; S recall is the weak spot because S looks like N and differs mainly in timing | Excellent |
| B. Multi-beat window | Window covering previous + current + next beat (e.g. 3×256 or ~2 s) so the CNN sees RR intervals implicitly | Better S recall | Still a plain 1D CNN; larger input, same ops |
| C. CNN + RR features | Beat window through CNN, RR-interval scalars concatenated before the FC layer | Best S recall in the literature | Awkward: accelerator wants one tensor input; can be done by appending RR values as extra input samples or channels, needs care in the YAML |
| D. Classical features + RF/SVM (baseline only) | Hand-crafted morphology + RR features | Reasonable, good baseline | Not a CNN; runs on the Arm core, not the accelerator; fine for §4.1 |
| E. Depthwise-separable / MobileNet-style 1D CNN | Fewer parameters for same depth | Same as A | Supported, but adds layers to describe in YAML; only worth it if size is tight (it is not: limit is 2 MB) |

Option B is the cheapest upgrade if S recall fails the success criterion.

### 4.3 Validation strategy inside DS1

| Option | Description | When to use |
|--------|-------------|-------------|
| **A. Fixed hold-out by record** (guide) | ~7 DS1 records as validation | Simple, fast, sufficient for the course |
| B. Leave-records-out k-fold (e.g. 5 folds over 22 records) | Every DS1 record is validated once; report mean ± std | More reliable for hyperparameter choice; 5× training cost |
| C. Leave-one-record-out | 22 folds | Overkill here |

### 4.4 Class imbalance

| Option | Simplicity | Notes |
|--------|-----------|-------|
| **Weighted cross-entropy** | One line | Start here |
| Weighted CE + oversampling of S/F | Sampler in `dataset.py` | Usually helps S/F recall a little |
| Focal loss | Small custom loss | Try if minority recall is still low |
| Undersampling N | Trivial | Throws away data; not recommended |
| Balanced batches | Sampler | Equivalent to oversampling in practice |
| No balancing | — | Model will predict N almost always; only useful to show in the report as a negative result |

### 4.5 Quantization path

| Option | Order | Notes |
|--------|-------|-------|
| **PTQ first, QAT if needed** | FP32 → `quantize.py` → evaluate; then QAT only if drop > threshold | Justifies QAT with numbers |
| QAT from the start | `--qat-policy` in training | Fine but skips the comparison the assignment asks for |
| Both reported | FP32 / PTQ-INT8 / QAT-INT8 columns | Best report, small extra cost |

### 4.6 Preprocessing

| Option | Notes |
|--------|-------|
| **Baseline removal (median filter or 0.5 Hz high-pass) + per-beat normalization + clipping** | Standard, cheap, works |
| No filtering, only normalization | Simpler; test it in the noise experiment to answer "does preprocessing help?" |
| Band-pass 0.5–40 Hz | Also removes muscle noise; check it does not distort QRS |
| Resampling to 250 Hz or 128 samples | Smaller input and model; MAX78002 can afford 256, so unnecessary |

Whatever is chosen must be feasible on the MAX78002 Arm core in C later. A
median filter or a first-order IIR high-pass is cheap; a long FIR is not.

### 4.7 Noise robustness experiment

| Option | Notes |
|--------|-------|
| **Add `bw`/`em`/`ma` noise to raw DS2 signals at 12/6/0 dB, then run the normal pipeline** | Matches the assignment exactly |
| Use only one noise type (e.g. `em`, the hardest) | Simpler, less informative |
| Mixed noise (sum of all three) | Closer to reality; report as an extra row |
| Also train with noise augmentation and compare | Answers "could robustness be improved?" with evidence |

### 4.8 Deployment evidence

| Option | Notes |
|--------|-------|
| **`ai8xize.py` output + logs + memory numbers from its verbose output** | Required minimum |
| Also build the generated project with the MSDK (`make`) | Proves the C compiles; still software-only, no board needed |
| Also run the known-answer test logic in a host-side simulation | Optional; ai8x provides `--simulate`-style checks in some versions |

---

## 5. Recommended simple path (updated)

```
Step 0–5   as in guide (setup, §1, download, EDA, prepare_data, verify splits)
Step 5b    NEW: toolchain smoke test — 1-epoch tiny CNN → quantize → synthesize
Step 6     Submission 1
Step 7     Baseline (RF or LR on validation)
Step 8     1D CNN written ONCE with ai8x layers; select by validation macro-F1
Step 9     PTQ first → evaluate INT8 → QAT only if needed → synthesize final model
Step 10–14 as in guide (DS2 evaluation, noise test, error analysis, write-up, package)
```

If validation S recall is below the success criterion after Step 8, switch
to the multi-beat window (4.2 B) before anything else. It is the one change
with the best cost/benefit ratio for this dataset.
