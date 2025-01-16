import yaml
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from ml_ops_project.model import load_model_config, initialize_model

# Load Model
model = initialize_model(load_model_config())

# Load Data


# Import Configuration
def load_training_config(
    config_path="/Users/frederikreimert/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/Kandidat_DTU/2024E/MLops/project_folder/ml_ops_project/configs/train/train_config.yaml",
):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Seq2SeqTrainingArguments(**config_dict)


training_args = load_training_config()

# Load Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_books["train"],
    eval_dataset=tokenized_books["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
