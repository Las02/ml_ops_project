import torch
import torch.nn.functional as F
import typer
import yaml
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import wandb
from ml_ops_project.data import OpusDataset
from ml_ops_project.evaluation import sacrebleu
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
    logger.warning(f"Using device: {device}")

    # Make dir for saving models if not exists
    Path("models/").mkdir(exist_ok=True)

    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # train_dataset = OpusDataset("data/test_data/test_data.txt")
    # test_dataset = OpusDataset("data/test_data/test_data.txt")
    # val_dataset = OpusDataset("data/test_data/test_data.txt")
    train_dataset = OpusDataset("data/processed/train.txt")
    test_dataset = OpusDataset("data/processed/test.txt")
    val_dataset = OpusDataset("data/raw/validation.txt")

    shuffle = True
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=shuffle)

    model.to(device)

    with wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        # track hyperparameters and run metadata
        config=train_config,
    ) as run:
        config = wandb.config
        for epoch in tqdm(range(config.epochs)):
            wandb.log(
                {
                    "epoch": epoch,
                }
            )
            logger.info(f"starting: epoch: {epoch}")
            train_epoch(model, optimizer, train_dataset, train_dataloader, device)
            # Apply model to test data
            test_val_epoch(
                model,
                optimizer,
                test_dataloader,
                loss_name="test_loss",
                test_dataset=test_dataset,
                device=device,
            )

        # Apply model to val data
        test_val_epoch(
            model,
            optimizer,
            val_dataloader,
            loss_name="val_loss",
            test_dataset=val_dataset,
            device=device,
        )

        torch.save(model.state_dict(), "models/model.pt")
        artifact = wandb.Artifact(
            name="ml_ops_project_model",
            type="model",
            description="A transformer model trained to translate text (DK-EN)",
            # metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
        )

        artifact.add_file("models/model.pt")
        run.log_artifact(artifact)


def train_epoch(model, optimizer, dataset, dataloader, device):
    for truth, input in dataloader:
        truth = truth.to(device)
        input = input.to(device)

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
        del truth
        del input


def test_val_epoch(model, optimizer, dataloader, loss_name, test_dataset, device):
    logger.info(f"Starting for {loss_name}")
    model.eval()
    total_loss = 0
    for truth, input in dataloader:
        truth = truth.to(device)
        input = input.to(device)

        outputs = model(input_ids=input, labels=truth)
        preds = F.softmax(outputs.logits, dim=-1).argmax(dim=-1)

        loss = outputs.loss
        total_loss += loss
        del truth
        del input
    total_loss = total_loss / len(test_dataset)

    logger.info(f"{loss_name} {total_loss}")
    wandb.log({f"{loss_name}": total_loss})
    bleu_score = sacrebleu(model, dataloader, test_dataset=test_dataset, batch_size=32)
    logger.info(f"bleu_score {bleu_score}")
    wandb.log({"bleu_score": bleu_score})

    model.train()


if __name__ == "__main__":
    app()
