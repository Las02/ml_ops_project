import token

import torch
import yaml
from fastapi import FastAPI
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import T5Tokenizer

from ml_ops_project.data import OpusDataset
from ml_ops_project.evaluation import sacrebleu
from ml_ops_project.model import initialize_model, load_model_config

app = FastAPI()


@app.post("/")
def main(input: str):
    ## Load the model configuration
    # config = load_model_config()
    #
    # # Reinitialize the model architecture
    # model = initialize_model(config)
    #
    # # Set the device
    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps"
    #     if torch.backends.mps.is_available()
    #     else "cpu"
    # )
    #
    # device = torch.device("cpu")  # Use CPU for validation
    #
    # # Load the saved state dictionary
    # state_dict = torch.load("models/model.pt", map_location=device)
    # model.load_state_dict(state_dict)
    #
    # # Move the model to the appropriate device
    # model.to(device)
    #
    # %%
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
    model = lambda x: f"{input} hello"

    # input = tokenizer(
    #     [input],
    #     return_tensors="pt",
    #     padding="do_not_pad",
    # ).input_ids
    # %%
    return model(input)

    if __name__ == "__main__":
        app()
