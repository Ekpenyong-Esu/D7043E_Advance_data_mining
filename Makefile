.PHONY: data prepare baseline train qat evaluate noise test

data:
	python -c "import wfdb; wfdb.dl_database('mitdb', 'data/raw/mitdb')"
	python -c "import wfdb; wfdb.dl_database('nstdb', 'data/raw/nstdb')"

prepare:
	python src/prepare_data.py --config configs/preprocess.yaml

baseline:
	python src/train.py --config configs/baseline.yaml

train:
	python src/train.py --config configs/cnn_fp32.yaml

qat:
	python src/train.py --config configs/cnn_qat.yaml

evaluate:
	python src/evaluate.py --config configs/cnn_qat.yaml --split test --final

noise:
	python src/noise_test.py --config configs/noise.yaml

test:
	pytest -q
