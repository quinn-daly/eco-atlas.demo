"""
embed.py — Embed chunks and store in ChromaDB

Reads processed JSON chunks from data/processed/
Embeds text via OpenAI text-embedding-3-small
Stores vectors + text + metadata in a local ChromaDB collection

Append behavior: chunks already in ChromaDB are skipped on reruns.
Safe to run multiple times — will only embed what's new.

Run: python pipeline/embed.py
"""

import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")

# ChromaDB stores its index here — committed to embeddings/ folder
CHROMA_DIR = Path("embeddings/chroma")

# ── Config ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "eco-materials"

EMBED_MODEL = "text-embedding-3-small"
# Same model used in ingest.py for semantic splitting — must stay consistent.
# If you change this, wipe ChromaDB and re-embed everything.

BATCH_SIZE = 100
# How many chunks to send to OpenAI per API call.
# 100 is safe within rate limits. Raise to 500 if you're on a higher tier.

# ── Clients (initialized once) ─────────────────────────────────────────────

openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
# PersistentClient saves to disk at CHROMA_DIR.
# Data survives between runs — this is what enables append behavior.

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
    # hnsw:space = cosine — measures angle between vectors, not magnitude.
    # Best choice for text: two chunks about the same topic will point in
    # the same direction even if one is longer than the other.
    # Must be set at collection creation — cannot change later without rebuilding.
)

# ── Core functions ──────────────────────────────────────────────────────────

def load_all_chunks() -> list[dict]:
    """Load every chunk from every JSON file in data/processed/."""
    chunks = []
    for json_file in sorted(PROCESSED_DIR.rglob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            chunks.extend(json.load(f))
    return chunks


def make_chunk_id(chunk: dict) -> str:
    """Generate a stable, unique ID for a chunk.

    Built from material_category + source filename + chunk_index.
    Same chunk always produces the same ID — this is how append mode
    knows what's already been embedded.
    """
    category = chunk["metadata"]["material_category"]
    source = chunk["metadata"]["source"]
    idx = chunk["metadata"]["chunk_index"]
    return f"{category}::{source}::{idx}"


def get_existing_ids() -> set[str]:
    """Fetch all chunk IDs already stored in ChromaDB.
    include=[] means: return IDs only, skip vectors and documents.
    Fast even on large collections.
    """
    result = collection.get(include=[])
    return set(result["ids"])


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Send a batch of texts to OpenAI. Returns one vector per text."""
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    # Response preserves input order — safe to zip directly with chunks.
    return [item.embedding for item in response.data]


def add_to_chroma(chunks: list[dict], vectors: list[list[float]], ids: list[str]):
    """Store a batch of chunks with their embeddings in ChromaDB."""
    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        return

    print("Loading chunks from data/processed/...")
    all_chunks = load_all_chunks()
    print(f"  {len(all_chunks)} total chunks loaded")

    print("Checking ChromaDB for already-embedded chunks...")
    existing_ids = get_existing_ids()
    print(f"  {len(existing_ids)} already embedded")

    new_chunks = [c for c in all_chunks if make_chunk_id(c) not in existing_ids]

    if not new_chunks:
        print("\nAll chunks already embedded. Nothing to do.")
        print(f"Collection '{COLLECTION_NAME}' has {collection.count()} chunks.")
        return

    print(f"  {len(new_chunks)} new chunks to embed\n")

    total_added = 0
    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ", flush=True)

        ids = [make_chunk_id(c) for c in batch]
        vectors = embed_batch([c["text"] for c in batch])
        add_to_chroma(batch, vectors, ids)

        total_added += len(batch)
        print("done")

    print(f"\nDone. {total_added} chunks embedded and stored.")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} total chunks.")


if __name__ == "__main__":
    main()
