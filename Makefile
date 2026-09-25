.PHONY: data prepare eda baseline train qat evaluate noise test report

# REQUESTS_CA_BUNDLE points at the system store so downloads work behind a corporate proxy
data: export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
data:
	.venv/bin/python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
	.venv/bin/python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"

prepare:
	.venv/bin/python src/prepare_data.py --config configs/preprocess.yaml

eda:
	.venv/bin/python src/make_eda_figures.py --config configs/preprocess.yaml

baseline:
	.venv/bin/python src/train.py --config configs/baseline.yaml

train:
	.venv/bin/python src/train.py --config configs/cnn_fp32.yaml

qat:
	.venv/bin/python src/train.py --config configs/cnn_qat.yaml

evaluate:
	.venv/bin/python src/evaluate.py --config configs/cnn_qat.yaml --split test --final

noise:
	.venv/bin/python src/noise_test.py --config configs/noise.yaml

test:
	.venv/bin/python -m pytest -q

report:
	cd report && pdflatex -interaction=nonstopmode report.tex && biber report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex && cp report.pdf ../report.pdf
