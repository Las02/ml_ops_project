import torch.nn.functional as F
import typer
import yaml
from loguru import logger

# %%
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from ml_ops_project.data import OpusDataset, Tokenize_data
from ml_ops_project.model import *
from ml_ops_project.model import initialize_model, load_model_config
from tqdm.auto import tqdm
import torch 
from torch.optim import AdamW

# Load Model
model = initialize_model(load_model_config())

# Set Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Set optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Load Data
app = typer.Typer()


@app.command()
def none():
    pass

@app.command()

def train(num_epochs: int = 10):
    config = load_model_config()
    model = initialize_model(config)

    dataset = OpusDataset("data/test_data/test_data.txt")
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    model.train()
    for epoch in tqdm(range(num_epochs)):
        for truth, input in train_dataloader:
            outputs = model(input_ids=input, labels=truth)
            preds = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            # Remove "<pad>" from preds
            preds_decoded = dataset.decode(preds)
            preds_decoded = [pred.replace("<pad>", "") for pred in preds_decoded]
            print(preds_decoded)

            logger.info("TRAIN SUCCESS")
            break


# %%

#
# # Import Configuration
# def load_training_config(config_path="/Users/frederikreimert/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Kandidat_DTU/2024E/MLops/project_folder/ml_ops_project/configs/train/train_config.yaml"):
#     with open(config_path, "r") as file:
#         config_dict = yaml.safe_load(file)
#     return Seq2SeqTrainingArguments(**config_dict)
#
# training_args = load_training_config()
#
# # Load Trainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_books["train"],
#     eval_dataset=tokenized_books["test"],
#     processing_class=tokenizer,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
# )


if __name__ == "__main__":
    app()
