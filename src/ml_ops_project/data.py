from pathlib import Path

import typer
from datasets import load_dataset
from torch.utils.data import Dataset

app = typer.Typer()

# class MyDataset(Dataset):
#    """My custom dataset."""
#
#    def __init__(self, raw_data_path: Path) -> None:
#        self.data_path = raw_data_path
#
#    def __len__(self) -> int:
#        """Return the length of the dataset."""
#
#    def __getitem__(self, index: int):
#        """Return a given sample from the dataset."""
#
#    def preprocess(self, output_folder: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""

# def preprocess(raw_data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MyDataset(raw_data_path)
#     dataset.preprocess(output_folder)


@app.command()
def download_data():
    ds = load_dataset("kaitchup/opus-Danish-to-English")

    with open("data/raw/train.txt", "w") as f:
        for line in ds["train"]["text"]:
            print(line, file=f)

    with open("data/raw/validation.txt", "w") as f:
        for line in ds["validation"]["text"]:
            print(line, file=f)


@app.command()
def do_nothing():
    pass


if __name__ == "__main__":
    app()
