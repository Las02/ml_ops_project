import torch.nn.functional as F

# %%
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from ml_ops_project.data import OpusDataset, Tokenize_data
from ml_ops_project.model import *

config = load_model_config()
model = initialize_model(config)

dataset = OpusDataset("data/test_data/test_data.txt")
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

for truth, input in train_dataloader:
    out = model(input_ids=input, labels=truth)
    preds = F.softmax(out.logits, dim=-1).argmax(dim=-1)

    # To translate back
    # dataset.decode(preds)
    break


# %%
