test_encode:
	pytest  tests/test_tokenize.py
download:
	python src/ml_ops_project/data.py download-data
split_data:
	python src/ml_ops_project/data.py preprocess
help:
	python src/ml_ops_project/data.py --help
test: 
	pytest

