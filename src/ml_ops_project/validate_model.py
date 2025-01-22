import torch
import yaml
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml_ops_project.data import OpusDataset
from ml_ops_project.evaluation import sacrebleu
from ml_ops_project.model import initialize_model, load_model_config

# Load the model configuration
config = load_model_config()

# Reinitialize the model architecture
model = initialize_model(config)

# Set the device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

device = torch.device("cpu")  # Use CPU for validation

# Load the saved state dictionary
state_dict = torch.load("models/model.pt", map_location=device)
model.load_state_dict(state_dict)

# Move the model to the appropriate device
model.to(device)
model.eval()

# Load training configurations
with open("configs/train/train_config.yaml", "r") as file:
    train_config = yaml.safe_load(file)
learning_rate = train_config["learning_rate"]

# Load test dataset
test_dataset = OpusDataset("data/processed/test.txt")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Initialize optimizer (though it's not needed for evaluation)
optimizer = AdamW(model.parameters(), lr=learning_rate)


# Test and evaluate the model
def test_val(model, dataloader, test_dataset, device):
    model.eval()
    total_loss = 0
    total_samples = 0

    # Iterate through the test dataset
    for truth, input in dataloader:
        # Move tensors to the correct device
        truth = truth.to(device, non_blocking=True)
        input = input.to(device, non_blocking=True)

        with torch.no_grad():
            # Get model outputs and calculate loss
            outputs = model(input_ids=input, labels=truth)
            loss = outputs.loss

            # Accumulate total loss, weighted by the number of samples in the batch
            batch_size = truth.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Calculate average loss over the entire test set
    average_loss = total_loss / total_samples
    logger.info(f"Test Loss: {average_loss}")

    # Compute BLEU score
    bleu_score = sacrebleu(model, dataloader, test_dataset=test_dataset, batch_size=32)
    logger.info(f"BLEU Score: {bleu_score}")

    # Return the metrics for further analysis or logging
    return average_loss, bleu_score


if __name__ == "__main__":
    test_val(model, test_dataloader, test_dataset, device)
