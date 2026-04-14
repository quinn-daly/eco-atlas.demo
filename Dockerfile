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

# Copy application code.
# .dockerignore excludes: .env, data/, app/, embeddings/, flashrank_cache/
# embeddings/ is intentionally excluded — it is fetched below via CHROMA_ARCHIVE_URL.
COPY . .

# ── Download ChromaDB from external archive ───────────────────────────────────
# The ChromaDB is 243 MB (too large for the Railway build-context upload).
# Instead we store the archive externally (e.g. a GitHub Release asset) and
# pull it here at image-build time so it is baked into the final image.
#
# How to set:
#   Railway dashboard → your service → Variables → Build Variables
#   Name : CHROMA_ARCHIVE_URL
#   Value: https://github.com/OWNER/REPO/releases/download/TAG/chroma_backup.tar.gz
#
# Archive format expected (create with the command in the manual steps):
#   embeddings/
#   embeddings/chroma/
#   embeddings/chroma/chroma.sqlite3
#   embeddings/chroma/<uuid>/data_level0.bin
#   embeddings/chroma/<uuid>/...
#
# When extracted at /app/ this produces /app/embeddings/chroma/ which matches
# the absolute path used by pipeline/query.py.
ARG CHROMA_ARCHIVE_URL
ARG CACHE_BUST=1
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && echo "Cache bust: $CACHE_BUST" \
 && curl -fL "$CHROMA_ARCHIVE_URL" -o chroma.tar.gz \
 && tar -xzf chroma.tar.gz \
 && rm chroma.tar.gz

# ── Build-time ChromaDB validation ───────────────────────────────────────────
# Fail the build immediately if the embeddings directory is missing or empty.
RUN python -c "\
import chromadb, sys, os; \
db_path = 'embeddings/chroma'; \
print('Validating ChromaDB at', os.path.abspath(db_path)); \
client = chromadb.PersistentClient(path=db_path); \
names = [c.name for c in client.list_collections()]; \
print('Collections present:', names); \
assert 'eco-materials' in names, \
    'DEPLOY FAILED: eco-materials collection not found in ' + str(names) + \
    '. The archive at CHROMA_ARCHIVE_URL may be incomplete or corrupted.'; \
count = client.get_collection('eco-materials').count(); \
print('eco-materials chunk count:', count); \
assert count > 0, 'DEPLOY FAILED: eco-materials collection is empty'; \
print('ChromaDB validation passed.')"

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
