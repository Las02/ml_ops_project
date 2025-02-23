import pickle
from pathlib import Path

import typer
from datasets import load_dataset
from loguru import logger
from tokenizers.normalizers import Lowercase, Replace, Sequence
from torch.utils.data import Dataset
from transformers import T5Tokenizer
import requests


app = typer.Typer()


class Tokenize_data:
    """Tokenize data"""

    return_tensors = "pt"
    padding = True

    def __init__(self, preprocess_data_path: Path, force_redownload: bool = False) -> None:
        self.data_path = preprocess_data_path
        self.danish, self.english = self.read_in_file(preprocess_data_path)

        stripped_path = self.data_path.strip("/")[-1]
        self.danish_cache_path = Path(f"data/cache/{stripped_path}_danish_data.pck")
        self.english_cache_path = Path(f"data/cache/{stripped_path}_english_data.pck")

        self.english_cache_path.parent.mkdir(exist_ok=True)
        self.danish_cache_path.parent.mkdir(exist_ok=True)

        # Set up tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
        # Normalize data
        normalizer = Sequence(
            [
                Replace("å", "aa"),
                Replace("ø", "oe"),
                Replace("æ", "ae"),
                Lowercase(),
            ]
        )
        self.tokenizer.normalizer = normalizer

        # Redownload paths if not exsist
        if (
            not Path(self.danish_cache_path).exists()
            or not Path(self.english_cache_path).exists()
            or force_redownload
        ):
            logger.warning(
                f"No cache containing tokenized data or force_redownload argument provided, begining tokenizing data: {preprocess_data_path}"
            )
            self.tokenize_data()
        else:
            logger.info(
                f"Cache found at {self.english_cache_path} and {self.danish_cache_path}: loading tokens from these"
            )
            with open(self.english_cache_path, "rb") as f:
                self.english_tokenized = pickle.load(f)
            with open(self.danish_cache_path, "rb") as f:
                self.danish_tokenized = pickle.load(f)

    def tokenize_data(self):
        logger.info("Starting tokenizing danish data")
        self.danish_tokenized = self.tokenizer(
            self.danish,
            return_tensors=self.return_tensors,
            padding=self.padding,
        )
        with open(self.danish_cache_path, "wb") as f:
            logger.info(f"Dumped danish pickle to {self.danish_cache_path}")
            pickle.dump(self.danish_tokenized, f)

        logger.info("Starting tokenizing english data")
        self.english_tokenized = self.tokenizer(
            self.english,
            return_tensors=self.return_tensors,
            padding=self.padding,
        )
        with open(self.english_cache_path, "wb") as f:
            logger.info(f"Dumped english pickle to {self.english_cache_path}")
            pickle.dump(self.english_tokenized, f)

    def read_in_file(self, preprocess_data_path: Path):
        logger.info(f"Reading in data from {preprocess_data_path}")
        # Read in the files and split them into danish and english
        danish_all = []
        english_all = []
        with open(preprocess_data_path, "r") as f:
            for line in f:
                line = line.split("###>")
                if len(line) != 2:
                    logger.critical(
                        f"line in dataset has more or less than two language entries: {line}"
                    )
                danish = line[0].strip()
                english = line[1].strip()
                danish_all.append(danish)
                english_all.append(english)
        return (danish_all, english_all)


# Preprocess by BM :.))
@app.command()
def split_data(
    raw_path: str = "data/raw",
    processed_path: str = "data/processed",
    test_percent: float = 0.2,
) -> None:
    """Split train.txt into processed/train.txt and processed/test.txt."""
    with open(f"{raw_path}/train.txt", "r") as f:
        lines = f.readlines()

    logger.info("Splitting downloaded dataset")
    n = len(lines)
    test_n = int(n * test_percent)

    with open(f"{processed_path}/train.txt", "w") as f:
        f.writelines(lines[test_n:])

    with open(f"{processed_path}/test.txt", "w") as f:
        f.writelines(lines[:test_n])


@app.command()
def tokenize_data():
    for path in [
        "data/test_data/test_data.txt",
        "data/processed/test.txt",
        "data/processed/train.txt",
        "data/raw/validation.txt",
    ]:
        logger.info(f"Begining data: {path}")
        Tokenize_data(path, force_redownload=True)


class OpusDataset(Dataset):
    """Opus dataset dataloader"""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.tokenize_data = Tokenize_data(self.data_path)
        self.len = len(self.tokenize_data.danish_tokenized["input_ids"])

    def decode(self, tokens: list[list]):
        return [self.tokenize_data.tokenizer.decode(x) for x in tokens]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        # return len(self.tokenize_data.danish)
        return self.len

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        input = self.tokenize_data.danish_tokenized["input_ids"][index]
        truth = self.tokenize_data.english_tokenized["input_ids"][index]
        return truth, input


@app.command()
def download_data():
    logger.info("Downloading dataset")
    ds = load_dataset("kaitchup/opus-Danish-to-English")

    # Mkdirs if not exisit to not crash
    Path("data/raw").mkdir(exist_ok=True)
    Path("data/processed").mkdir(exist_ok=True)

    with open("data/raw/train.txt", "w") as f:
        for line in ds["train"]["text"]:
            print(line, file=f)

    with open("data/raw/validation.txt", "w") as f:
        for line in ds["validation"]["text"]:
            print(line, file=f)


@app.command()
def download_public_gcs_file(object_name: str, destination_file_name: str):
    """Downloads a public file from Google Cloud Storage."""

    url = f"https://storage.googleapis.com/data_opus_dk_en/{object_name}"

    # Download object from url
    response = requests.get(url)
    response.raise_for_status()

    # Save object to file
    with open(destination_file_name, "wb") as file:
        file.write(response.content)

    print(f"File downloaded to {destination_file_name}.")


if __name__ == "__main__":
    app()
