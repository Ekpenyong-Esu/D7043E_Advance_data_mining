# How to build the project with the Makefile

## Understanding the problem

The project teaches a small model to look at one heartbeat and decide which of four types it is. These are the standard AAMI classes:

| Class | Name | What happens in the heart |
|---|---|---|
| **N** | Normal beat | The beat starts in the SA node (the heart's natural pacemaker) and travels the normal path. Some harmless variations, such as bundle branch blocks, are counted here too. |
| **S** | Supraventricular ectopic beat | An early beat that starts in the upper part of the heart (the atria), but not at the SA node. It looks almost normal, but it comes too early. |
| **V** | Ventricular ectopic beat | An early beat that starts in the lower chambers (the ventricles). It skips the normal path, so the QRS is wide and oddly shaped. This is the most important one to catch. |
| **F** | Fusion beat | A normal beat and a V beat happen at the same moment and merge into one. Its shape is halfway between N and V. It is very rare. |

Paced beats (from a pacemaker) and unknown beats are left out.

---

The Makefile has 5 commands. Run them from the project folder, in this order:

```bash
make data
make prepare
make eda
make test
make report
```

Before the first run, the `.venv` folder must exist, because every command uses `.venv/bin/python`.

---

## 1. `make data`

```makefile
data: export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
data:
	.venv/bin/python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
	.venv/bin/python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"
```

**What it does:** downloads the two ECG databases into `data/raw/`.
- `mitdb`: the heart recordings, with every beat labelled.
- `nstdb`: real ECG noise.

The first line points Python at the system certificates, so the download doesn't fail behind a company network.

**Why first:** every other step needs these files. You only run it once.

---

## 2. `make prepare`

```makefile
prepare:
	.venv/bin/python src/prepare_data.py --config configs/preprocess.yaml
```

**What it does:** turns the raw recordings into clean heartbeats, using the settings in `configs/preprocess.yaml`. Results go to `data/processed/` (train, val and test) and `results/`.

**Why second:** it needs the raw data from step 1, and the next steps need its output.

---

## 3. `make eda`

```makefile
eda:
	.venv/bin/python src/make_eda_figures.py --config configs/preprocess.yaml
```

**What it does:** makes the figures and tables that describe the data. They go to `report/figures/` and `results/eda/`.

**Why third:** one of the figures uses a prepared heartbeat, so step 2 must run first.

---

## 4. `make test`

```makefile
test:
	.venv/bin/python -m pytest -q
```

**What it does:** runs the tests in `tests/`. `-q` keeps the output short.

**Why here:** it checks the code is correct before you write the results into the report.

---

## 5. `make report`

```makefile
report:
	cd report && pdflatex ... && biber report && pdflatex ... && pdflatex ... && cp report.pdf ../report.pdf
```

**What it does:** builds the report PDF and copies it to `report.pdf` in the project folder.
- `pdflatex` turns the LaTeX into a PDF.
- `biber` adds the references.
- `pdflatex` runs twice more so the page numbers, table of contents and citations come out right.

**Why last:** the report uses the figures and tables made in steps 2 and 3.
