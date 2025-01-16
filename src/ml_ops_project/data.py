from pathlib import Path

import torch
import typer
from datasets import load_dataset
from loguru import logger
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("t5-small")

app = typer.Typer()


class Tokenize_data:
    """Tokenize data"""

    return_tensors = "pt"
    padding = True

    def __init__(self, preprocess_data_path: Path) -> None:
        self.data_path = preprocess_data_path
        self.danish, self.english = self.read_in_file(preprocess_data_path)
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        self.danish_tokenized = self.tokenizer(self.danish, return_tensors=self.return_tensors, padding=self.padding)
        # breakpoint()
        self.english_tokenized = self.tokenizer(self.english, return_tensors=self.return_tensors, padding=self.padding)

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


# from transformers import T5Tokenizer, T5ForConditionalGeneration
# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
# tokenizer("translate English to German: How are you?”)’’’
#
# ‘’’
# {'input_ids': [13959, 1566, 12, 2968, 10, 571, 33, 25, 58, 1], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}


# Preprocess by BM :.))
@app.command()
def split_data(raw_path: str = "data/raw", processed_path: str = "data/processed", test_percent: float = 0.2) -> None:
    """Split train.txt into processed/train.txt and processed/test.txt."""
    with open(f"{raw_path}/train.txt", "r") as f:
        lines = f.readlines()

    n = len(lines)
    test_n = int(n * test_percent)

    with open(f"{processed_path}/train.txt", "w") as f:
        f.writelines(lines[test_n:])

    with open(f"{processed_path}/test.txt", "w") as f:
        f.writelines(lines[:test_n])


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
