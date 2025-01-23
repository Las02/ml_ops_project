# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN apt install make

COPY src src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml
COPY Makefile Makefile
COPY models models/




# RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose


EXPOSE 8080
ENTRYPOINT ["fastapi", "run", "src/ml_ops_project/api.py", "--host", "0.0.0.0", "--port", "8080"]
# ENTRYPOINT ["uvicorn", "src/ml_ops_project/api:app", "--host", "0.0.0.0", "--port", "8000"]

