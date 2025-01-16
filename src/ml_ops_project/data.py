from pathlib import Path

import torch
import typer
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("t5-small")

app = typer.Typer()


class Tokenize_data:
    """preprocess data"""

    def __init__(self, preprocess_data_path: Path) -> None:
        self.data_path = preprocess_data_path
        self.danish, self.english = self.read_in_file(preprocess_data_path)

    def read_in_file(self, preprocess_data_path: Path):
        danish_all = []
        english_all = []
        with open(preprocess_data_path, "r") as f:
            for line in f:
                line = line.split("###>")
                if len(line) != 2:
                    logger.critical(f"line in dataset has more or less than two language entries: {line}")
                danish = line[0].strip()
                english = line[1].strip()
                danish_all.append(danish)
                english_all.append(english)
        return (danish_all, english_all)


# Preprocess by BM :.))
@app.command()
def preprocess(test_percent: float = 0.2) -> None:
    """Split train.txt into preprocessed/train.txt and preprocessed/test.txt."""
    with open("data/raw/train.txt", "r") as f:
        lines = f.readlines()

    n = len(lines)
    test_n = int(n * test_percent)

    with open("data/processed/train.txt", "w") as f:
        for line in lines[test_n:]:
            print(line, file=f)

    with open("data/processed/test.txt", "w") as f:
        for line in lines[:test_n]:
            print(line, file=f)


@app.command()
def download_data():
    ds = load_dataset("kaitchup/opus-Danish-to-English")

    with open("data/raw/train.txt", "w") as f:
        for line in ds["train"]["text"]:
            print(line, file=f)

    with open("data/raw/validation.txt", "w") as f:
        for line in ds["validation"]["text"]:
            print(line, file=f)


if __name__ == "__main__":
    app()

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
