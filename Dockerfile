FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only (no streamlit / langchain / pymupdf)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Pre-download FlashRank reranker model at build time.
# Stored in /app/flashrank_cache (not /tmp) so it survives container restarts
# — /tmp can be ephemeral on Railway and similar platforms.
RUN python -c "\
from flashrank import Ranker; \
Ranker(model_name='ms-marco-MiniLM-L-12-v2', cache_dir='/app/flashrank_cache')"

# Copy application code and the ChromaDB index.
# .dockerignore excludes: .env, data/ (raw PDFs + processed JSON), app/ (Streamlit UI)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
