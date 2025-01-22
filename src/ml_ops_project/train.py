import torch
import torch.nn.functional as F
import typer
import wandb
from loguru import logger
from torch.optim import AdamW
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from ml_ops_project.data import OpusDataset
from ml_ops_project.model import *
from ml_ops_project.model import initialize_model, load_model_config

# Load Data
app = typer.Typer()

@app.command()
def none():
    pass

@app.command()
def train():
    with open("configs/train/train_config.yaml", "r") as file:
        train_config = yaml.safe_load(file)

    num_epochs = train_config["epochs"]
    learning_rate = train_config["learning_rate"]

    logger.info("Starting training")

    config = load_model_config()
    model = initialize_model(config)

    # Set Device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Make dir for saving models if not exists
    Path("models/").mkdir(exist_ok=True)

    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # dataset = OpusDataset("data/processed/train.txt")
    dataset = OpusDataset("data/test_data/test_data.txt")
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

    with wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "epochs": num_epochs,
        },
    ) as run:
        config = wandb.config
        for epoch in tqdm(range(config.epochs)):
            wandb.log(
                {
                    "epoch": epoch,
                }
            )
            logger.info(f"starting: epoch: {epoch}")
            train_epoch(model, optimizer, dataset, train_dataloader)

        torch.save(model.state_dict(), "models/model.pt")
        artifact = wandb.Artifact(
            name="ml_ops_project_model",
            type="model",
            description="A transformer model trained to translate text (DK-EN)",
            # metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
        )

        artifact.add_file("models/model.pt")
        run.log_artifact(artifact)


def train_epoch(model, optimizer, dataset, dataloader):
    for truth, input in dataloader:
        outputs = model(input_ids=input, labels=truth)
        preds = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # Remove "<pad>" from preds
        preds_decoded = dataset.decode(preds)
        preds_decoded = [pred.replace("<pad>", "") for pred in preds_decoded]

        logger.info(f"loss {loss}")
        wandb.log({"loss": loss})


if __name__ == "__main__":
    app()
