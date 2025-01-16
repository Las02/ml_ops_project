import yaml
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from ml_ops_project.model import load_model_config, initialize_model
from ml_ops_project.data import Tokenize_data, OpusDataset
from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader

# Load Model
model = initialize_model(load_model_config())

# Load Data
dataset = OpusDataset("data/test_data/test_data.txt")
train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Tokenize Data
tokenize_data = Tokenize_data("data/test_data/test_data.txt")

print(dataset)
print(train_dataloader)
print(tokenize_data.danish[0])
print(tokenize_data.english[0])


# Import Configuration
def load_training_config(config_path="/Users/frederikreimert/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Kandidat_DTU/2024E/MLops/project_folder/ml_ops_project/configs/train/train_config.yaml"):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Seq2SeqTrainingArguments(**config_dict)

training_args = load_training_config()

# Load Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataloader,
    eval_dataset=,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


