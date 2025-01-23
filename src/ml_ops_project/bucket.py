from google.cloud import storage
import typer

app = typer.Typer()


@app.command()
def upload_blob(
    bucket_name: str = typer.Option("mlops-models-2025", help="Name of the GCS bucket"),
    source_file_name: str = typer.Option(r"models/model.pt", help="Path to the local file"),
    destination_blob_name: str = typer.Option(
        "model-from-upload.pt", help="Destination blob name in GCS"
    ),
):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)

        # Optional: set a generation-match precondition to avoid race conditions
        generation_match_precondition = 0

        blob.upload_from_filename(
            source_file_name, if_generation_match=generation_match_precondition
        )

        print(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        print(f"An error occurred: {e}")


@app.command()
def download_blob(
    bucket_name: str = typer.Option("mlops-models-2025", help="Name of the GCS bucket"),
    source_blob_name: str = typer.Option("model-from-upload", help="Name of the blob in GCS"),
    destination_file_name: str = typer.Option(
        r"models/downloaded_model.pt", help="Path to save the downloaded file"
    ),
):
    """Downloads a blob from the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)

        blob.download_to_filename(destination_file_name)

        print(
            f"Downloaded storage object {source_blob_name} from bucket {bucket_name} to local file {destination_file_name}."
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    app()
