FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==2.3.2

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (no dev deps, no root package)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main

# Copy source code
COPY . .

# Create data directories
RUN mkdir -p data/chroma_db data/uploads data/sample_contracts data/synthetic_eval logs

# Expose ports
EXPOSE 8000 8501

# Default command (can be overridden in docker-compose)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
