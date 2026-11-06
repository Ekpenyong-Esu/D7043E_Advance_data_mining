.PHONY: run eda model test report toolchain

# Two notebooks run everything. Submission 1: notebooks/EDA.ipynb (download, preprocessing, EDA, report sections 1 to 3).
# Submission 2: notebooks/MODELING.ipynb (baselines, model selection, CNN, quantisation, DS2 evaluation, noise test,
# error analysis, MAX78002 synthesis, report sections 4 to 8). `make run` executes both headless and saves their
# outputs in place. Open them in Jupyter to do the same by hand.
run: eda model

eda:
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 EDA.ipynb

model:
	cd notebooks && ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 MODELING.ipynb

# Shortcuts for the last two notebook steps, for when the code or the LaTeX changed but the data did not.
test:
	.venv/bin/python -m pytest -q

report:
	cd report && pdflatex -interaction=nonstopmode report.tex && biber report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex && cp report.pdf ../report.pdf

# The Analog Devices toolchain, needed by MODELING.ipynb for quantisation and synthesis. Cloned into toolchain/ (git-ignored).
# ai8x-training only provides ai8x.py (imported into .venv). ai8x-synthesis runs in its own venv with the versions it pins.
toolchain:
	mkdir -p toolchain
	test -d toolchain/ai8x-training || git clone --depth 1 https://github.com/analogdevicesinc/ai8x-training.git toolchain/ai8x-training
	test -d toolchain/ai8x-synthesis || git clone --depth 1 https://github.com/analogdevicesinc/ai8x-synthesis.git toolchain/ai8x-synthesis
	test -x toolchain/ai8x-synthesis/venv/bin/python || uv venv toolchain/ai8x-synthesis/venv --python 3.11
	uv pip install --python toolchain/ai8x-synthesis/venv/bin/python torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python toolchain/ai8x-synthesis/venv/bin/python -r toolchain/ai8x-synthesis/requirements.txt
