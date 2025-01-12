from pathlib import Path

import torch
import typer
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

app = typer.Typer()


class preprocess_data:
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


# dl = Opus_datalaoder("./data/raw/train.txt")
# dl.danish
# %%


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
