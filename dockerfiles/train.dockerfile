# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/

# Set working directory
WORKDIR /

# Install dependencies
#RUN pip install -r requirements.txt --no-cache-dir
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir

# Entrypoint
ENTRYPOINT ["python", "-u", "src/ml_ops_project/train.py"]


