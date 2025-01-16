import subprocess

import pytest
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer

from ml_ops_project.model import initialize_model, load_model_config


def test_config_loads():
    """
    Ensures that the configuration loads correctly and returns a T5Config instance.
    """
    config = load_model_config()  # loads configs/model/model_config.yaml
    assert isinstance(config, T5Config), "load_model_config did not return a T5Config object."


def test_initialize_model():
    """
    Tests whether the model initializes properly from the given T5Config.
    """
    config = load_model_config()
    model = initialize_model(config)
    assert isinstance(
        model, T5ForConditionalGeneration
    ), "initialize_model did not return a T5ForConditionalGeneration model."


def test_model_script_runs():
    """
    Runs the model.py script in a subprocess to ensure it completes without errors.
    """
    result = subprocess.run(["python", "src/ml_ops_project/model.py"], capture_output=True, text=True)
    assert result.returncode == 0, f"model.py failed to run:\n{result.stderr}"
