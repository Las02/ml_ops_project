# evaluation.py
import typer
from loguru import logger
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn.functional as F
import evaluate
import numpy as np
import tqdm as tqdm
from ml_ops_project.data import OpusDataset
from ml_ops_project.model import load_model_config
from transformers import T5ForConditionalGeneration

app = typer.Typer()

logger.info("Loading evaluation method")
bleu_metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

@app.command()
def evaluate_model(
    data_path: str = "data/test_data/test_data.txt",
    model_path: str = "models/model.pt",
    batch_size: int = 2
):
    logger.info("Loading model config")
    model = torch.load("models/torch_model.pt", map_location="cpu")
    model.eval()

    logger.info("Loading test data")
    dataset = OpusDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    logger.info("Evaluating model")

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (truth_ids, input_ids) in enumerate(dataloader):

            generated_tokens = model.generate(
                input_ids=input_ids,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

            batch_preds = dataset.decode(generated_tokens)

            batch_truths = dataset.decode(truth_ids)

            # Clean up <pad> tokens
            batch_preds = [p.replace("<pad>", "").strip() for p in batch_preds]
            batch_truths = [t.replace("<pad>", "").strip() for t in batch_truths]

            all_predictions.extend(batch_preds)
            all_targets.extend(batch_truths)

            logger.info(f"Batch {batch_idx}:")
            for p, r in zip(batch_preds, batch_truths):
                logger.info(f"Predicted: {p} | Reference: {r}")

    logger.info("Post-processing text for BLEU computation...")
    final_preds, final_refs = postprocess_text(all_predictions, all_targets)

    logger.info("Computing BLEU score with SacreBLEU...")
    bleu_score = bleu_metric.compute(predictions=final_preds, references=final_refs)
    logger.info(f"BLEU: {bleu_score['score']:.4f}")
    print(f"\nFinal BLEU score: {bleu_score['score']:.4f}\n")

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    app()