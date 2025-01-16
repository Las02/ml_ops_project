import yaml
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration

# Import Configuration
def load_model_config(config_path="configs/t5_small_config.yaml"):
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return T5Config(**config_dict)

# Load model
def initialize_model(config):
    model = T5ForConditionalGeneration(config)
    model.from_pretrained("google-t5/t5-small")
    return model

if __name__ == "__main__":
    model = initialize_model(load_model_config())