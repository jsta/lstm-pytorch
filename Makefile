
all: data

data: data/data.ardata data/lstm-baseline_model.pytorch

data/data.ardata: scripts/01_generate_data.py scripts/generate_data_utils.py
	python scripts/01_generate_data.py

data/lstm-baseline_model.pytorch: scripts/02_train.py data/data.ardata
	python scripts/02_train.py

predict.png: scripts/03_plot.py
	cd scripts && python 03_plot.py