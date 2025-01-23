import torch
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from tokenizers.normalizers import Sequence, Replace, Lowercase

def translate_danish_to_english(input_text: str):

    #Load the base T5 model
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

    #Load your custom fine-tuned weights
    model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
    model.eval()

    #Load the matching tokenizer
    tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")

    normalizer = Sequence([
        Replace("å", "aa"),
        Replace("ø", "oe"),
        Replace("æ", "ae"),
        Lowercase(),
    ])
    tokenizer.normalizer = normalizer

    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = tokenizer.pad_token_id

    #Build a pipeline for translation da->en
    translator = pipeline(
        task="translation_da_to_en",
        model=model,
        tokenizer=tokenizer,
        device=-1,  # run on CPU
    )

    #Translate
    result = translator(input_text, max_length=240)

    # result is a list of dicts [{'translation_text': '...'}]
    return result[0]["translation_text"]