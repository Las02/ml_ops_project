test_encode:
	# Test only the encoding part
	# !! Might not work
	python -m pytest  tests/test_tokenize.py --capture=no -vv
setup_data:
	# download data and split it
	python src/ml_ops_project/data.py download-data
	python src/ml_ops_project/data.py split-data
	python src/ml_ops_project/data.py tokenize-data
test: 
	# Run all tests
	python -m pytest
train:
	# Train model
	python  src/ml_ops_project/train.py train
build_cloud:
	gcloud builds submit --config=cloudbuild.yaml
download_model:
	python src/ml_ops_project/bucket.py download-blob
