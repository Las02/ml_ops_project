# evaluation.py
import typer
from loguru import logger
import torch
import evaluate

app = typer.Typer()

logger.info("Loading evaluation method")
bleu_metric = evaluate.load("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


@app.command()
def sacrebleu(model,
                   test_dataloader, 
                   test_dataset,
    batch_size: int = 2): 

    logger.info("Evaluating model")
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (truth_ids, input_ids) in enumerate(test_dataloader):
            generated_tokens = model.generate(
                input_ids=input_ids, max_length=64, num_beams=4, early_stopping=True
            )

            batch_preds = test_dataset.decode(generated_tokens)

            batch_truths = test_dataset.decode(truth_ids)

            # Clean up <pad> tokens
            batch_preds = [p.replace("<pad>", "").strip() for p in batch_preds]
            batch_truths = [t.replace("<pad>", "").strip() for t in batch_truths]

            all_predictions.extend(batch_preds)
            all_targets.extend(batch_truths)

            logger.info(f"Batch {batch_idx}:")
            for p, r in zip(batch_preds, batch_truths):
                logger.info(f"Predicted: {p} | Reference: {r}")

    final_preds, final_refs = postprocess_text(all_predictions, all_targets)

    logger.info("Computing BLEU score with SacreBLEU...")
    bleu_score = bleu_metric.compute(predictions=final_preds, references=final_refs)
    logger.info(f"BLEU: {bleu_score['score']:.4f}")

    logger.info("Evaluation complete!")
