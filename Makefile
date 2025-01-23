test_encode:
	# Test only the encoding part
	# !! Might not work
	python -m pytest  tests/test_tokenize.py --capture=no -vv
setup_data:
	# download data and split it
	python src/ml_ops_project/data.py download-data
	python src/ml_ops_project/data.py split-data
	python src/ml_ops_project/data.py tokenize-data
setup_data_from_cloud:
	# download data and split it
	python src/ml_ops_project/data.py download-public-gcs-file train.txt data/raw/train.txt
	python src/ml_ops_project/data.py download-public-gcs-file validation.txt data/raw/validation.txt
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
run_streamlit:
	python -m streamlit run src/ml_ops_project/streamlit.py 
start_backend_and_frontend:
	fastapi run --reload src/ml_ops_project/api.py
	python -m streamlit run src/ml_ops_project/streamlit.py 
build_docker_api:
	sudo docker build -t api . -f dockerfiles/api.dockerfile
run_docker_api_locally:
	sudo docker run --rm --name my_fastapi_app -p 8000:8000 api
api_build_cloud:
	gcloud builds submit --config=api_cloudbuild.yaml
download_model:
	python src/ml_ops_project/bucket.py download-blob --bucket-name "mlops-models-2025" --source-blob-name "model.pt" --destination-file-name "models/model.pt"
