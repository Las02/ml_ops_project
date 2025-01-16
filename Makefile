test_encode:
	pytest  tests/test_tokenize.py --capture=no -vv
setup_data:
	# download data and split it
	python src/ml_ops_project/data.py download-data
	python src/ml_ops_project/data.py split-data
help:
	python src/ml_ops_project/data.py --help
test: 
	pytest --capture=no -vv

