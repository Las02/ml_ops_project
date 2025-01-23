from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI
from tokenizers.normalizers import Lowercase, Replace, Sequence
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

from ml_ops_project.inference import translate_danish_to_english


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # Load the base T5 model
    app.model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    # Load your custom fine-tuned weights
    app.model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    app.model.eval()

    yield
    # Shutdown
    del app.state.model


app = FastAPI(lifespan=lifespan)


@app.post("/")
def main(input: str):
    return translate_danish_to_english(app.model, input)

    # model = lambda x: x[::-1]
    # return model(input)


if __name__ == "__main__":
    app()
