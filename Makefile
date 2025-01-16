download:
	python src/ml_ops_project/data.py download-data
split_data:
	python src/ml_ops_project/data.py preprocess
help:
	python src/ml_ops_project/data.py --help
test: 
	pytest

