import subprocess
import pytest
from transformers import T5Tokenizer, T5ForConditionalGeneration

def test_model_script_runs():
    result = subprocess.run(["python", "src/ml_ops_project/model.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"model.py failed to run: {result.stderr}"