#  Project description

This project focuses on developing a machine learning translation model for English-to-Danish language translation by fine-tuning a pre-trained T5 (Text-to-Text Transfer Transformer) model. 
The T5 models are encoder-decoder models developed by Google. Unlike many models that specialize in specific tasks, T5 reframes all NLP tasks—such as translation, summarization, and question answering—as a text-to-text problem. This means both the input and output are always treated as text strings, making it highly adaptable across tasks. 
We will use the OPUS project's data for fine-tuning the model. The OPUS project is an open source project providing translated text from the internet. Here we will use the subset of the dataset containg English-to-Danish translations. This dataset consists of 52 million sentences which tokenized results in about 700 million danish and english tokens. Due to the size of the dataset we will start by working with a subsample of the dataset, and ideally later expand to the full dataset if time allows for it.
We intend to use the Hugging Face Transformers framework to streamline the model training pipeline. The Hugging face Transformers framework provides tools to more easily work with pretrained transformers model, for tasks such as NLP, Audio classification and translation. The Transformers framework work with various deep learning frameworks including Pytorch which we plan on using for this project.  
While one of our goals of the project is to produce a model which is capable of basic translation from English-to-Danish, our main goal in this project is the Machine Learning Operations aspect of developing the model. Meaning that our metric of success is not the performance of the model but rather the quality of the code, the reproducibility of our experiments and how well we have utilized the tools learned in the course. This involves using version control systems to organize our progress during the group work, and being able to efficiently implement machine learning models locally and in the cloud. 


# Installation
```
conda create -n ml_ops 'python==3.12.2'
conda activate ml_ops
pip install -e .
```

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
