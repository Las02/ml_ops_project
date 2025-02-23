# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src src/
COPY requirements.txt requirements.txt
COPY frontend_requirements.txt frontend_requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN --mount=type=cache,target=/root/.cache/pip pip install -r frontend_requirements.txt
RUN pip install . --no-deps --no-cache-dir --verbose

EXPOSE 8080
ENTRYPOINT ["python", "-m", "streamlit", "run", "src/ml_ops_project/streamlit.py", "--server.port", "8080"]
# ENTRYPOINT ["uvicorn", "src/ml_ops_project/api:app", "--host", "0.0.0.0", "--port", "8000"]


