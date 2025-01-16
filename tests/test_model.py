import subprocess
import pytest
from transformers import T5Tokenizer, T5ForConditionalGeneration

def test_model_script_runs():
    result = subprocess.run(["python", "src/ml_ops_project/model.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"model.py failed to run: {result.stderr}"

@pytest.fixture
def model():
    model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
    return model

@pytest.fixture
def tokenizer():
    return T5Tokenizer.from_pretrained("google-t5/t5-small")

def test_simple_example(model, tokenizer):
    input_text = "translate English to Danish: Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt")
    
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    assert isinstance(decoded_output, str), "Model output is not a string."
    assert len(decoded_output) > 0, "Model output is empty."
    print(f"Model output: {decoded_output}")
