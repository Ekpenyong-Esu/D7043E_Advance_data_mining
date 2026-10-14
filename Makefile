.PHONY: data prepare test report

# REQUESTS_CA_BUNDLE points at the system store so downloads work behind a corporate proxy
data: export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
data:
	.venv/bin/python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
	.venv/bin/python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"

prepare:
	.venv/bin/python src/prepare_data.py --config configs/preprocess.yaml

test:
	.venv/bin/python -m pytest -q

report:
	cd report && pdflatex -interaction=nonstopmode report.tex && biber report && pdflatex -interaction=nonstopmode report.tex && pdflatex -interaction=nonstopmode report.tex && cp report.pdf ../report.pdf
