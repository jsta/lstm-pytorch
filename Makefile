
all: data

data: data/data.ardata data/lstm-baseline_model.pytorch

data/data.ardata: scripts/01_generate_data.py scripts/99_generate_data_utils.py
	cd scripts && python 01_generate_data.py

data/lstm-baseline_model.pytorch: scripts/02_train.py data/data.ardata
	cd scripts && python 02_train.py