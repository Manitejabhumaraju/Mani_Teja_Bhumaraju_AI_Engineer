# Dockerfile for the Loadshare RCA Agent (Python services)
#
# Single image that can run either the FastAPI backend or the Streamlit
# frontend depending on the `command:` in docker-compose. Keeps the image
# list short and the build cache useful.
#
# The MCP server has its own Dockerfile under mcp-servers/Dockerfile.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps — kept minimal. curl for healthchecks only.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so cold-start doesn't fetch from HuggingFace.
# Bakes ~80MB into the image but makes demo-day deterministic.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    || echo "warning: embedding model preload skipped"

# Application code (changes most often → copied last for cache efficiency)
COPY app       ./app
COPY frontend  ./frontend
COPY scripts   ./scripts
COPY docs      ./docs
COPY data      ./data
COPY eval      ./eval
COPY README.md ./

# Build the DB at image-build time so the container is ready to serve.
RUN python scripts/load_csv_to_sqlite.py || echo "DB build deferred to runtime"

EXPOSE 8000 8501

# Healthcheck for the backend service. Streamlit frontend overrides via compose.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# Default command runs the backend. The frontend container overrides this
# via the `command:` field in docker-compose.yml.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
