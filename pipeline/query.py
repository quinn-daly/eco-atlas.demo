"""
query.py — Retrieval and response pipeline

Given a user question:
1. Embeds the question with text-embedding-3-small
2. Retrieves the top-k most relevant chunks from ChromaDB
3. Passes retrieved chunks + question to GPT-4o
4. Returns a grounded answer with source citations

Can be imported as a module (from app layer) or run as an interactive CLI.

Run: python pipeline/query.py
"""

import os
import sys
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "eco-materials"
EMBED_MODEL = "text-embedding-3-small"
# Must match the model used in ingest.py and embed.py — never change independently.

CHAT_MODEL = "gpt-4o"

TOP_K = 6
# Number of chunks retrieved per query.
# More = richer context for GPT-4o, but higher token cost and more noise risk.
# 6 is a good starting point for dense academic material — tune after testing.

TOP_K_PER_MATERIAL = 4
# For comparison queries, how many chunks to fetch per detected material.
# Each named material gets its own filtered retrieval, so both are guaranteed
# representation regardless of how comprehensive references rank.

TOP_K_GENERAL = 4
# Additional unfiltered chunks fetched alongside per-material results
# to catch cross-material sources (e.g. a paper comparing both materials).

MAX_CHUNKS_PER_AUTHOR = 1
# Max chunks kept from any single author after retrieval.
# Prevents one comprehensive reference (e.g. Almusaed, Hanaor, Chen) from
# dominating all slots across its many topic-labeled copies.

# ── Material detection ──────────────────────────────────────────────────────
# Maps lowercase query terms to exact ChromaDB material_category values.
# Add aliases here if new materials are added to the knowledge base.

MATERIAL_ALIASES: dict[str, str] = {
    "cork": "Cork",
    "sheep wool": "Sheep Wool",
    "wool": "Sheep Wool",
    "bamboo": "Bamboo",
    "hemp": "Hemp",
    "hempcrete": "Concrete (Hempcrete, Ashcrete, Timbercrete, Ferrock)",
    "ashcrete": "Concrete (Hempcrete, Ashcrete, Timbercrete, Ferrock)",
    "timbercrete": "Concrete (Hempcrete, Ashcrete, Timbercrete, Ferrock)",
    "ferrock": "Concrete (Hempcrete, Ashcrete, Timbercrete, Ferrock)",
    "mycelium": "Mycellium",
    "mycelium": "Mycellium",
    "mushroom": "Mycellium",
    "cob": "Cob",
    "cellulose": "Cellulose",
    "linoleum": "Linoleum",
    "straw": "Wheat Straw - Straw Bales",
    "wheat straw": "Wheat Straw - Straw Bales",
    "straw bale": "Wheat Straw - Straw Bales",
    "rammed earth": "Compacted Soil (Rammed earth, Earth bags)",
    "earth bag": "Compacted Soil (Rammed earth, Earth bags)",
    "compacted soil": "Compacted Soil (Rammed earth, Earth bags)",
    "solar shingle": "Solar Shingles",
    "recycled plastic": "Recycled Plastic + Rubber",
    "rubber": "Recycled Plastic + Rubber",
    "jute": "Natural Plant Fibers (flax, coconut, kenaf, jute)",
    "flax": "Natural Plant Fibers (flax, coconut, kenaf, jute)",
    "kenaf": "Natural Plant Fibers (flax, coconut, kenaf, jute)",
    "coconut": "Natural Plant Fibers (flax, coconut, kenaf, jute)",
    "natural fiber": "Natural Plant Fibers (flax, coconut, kenaf, jute)",
    "plant fiber": "Natural Plant Fibers (flax, coconut, kenaf, jute)",
    "clay plaster": "Bio Clay Plaster",
    "bio clay": "Bio Clay Plaster",
    "polyurethane foam": "Plant-based rigid polyurethane foam",
    "pu foam": "Plant-based rigid polyurethane foam",
}

CHROMA_DIR = Path("embeddings/chroma")

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert assistant for a sustainable building materials knowledge base.
Students use you to research materials, compare properties, and plan builds.

Answer questions using ONLY the source excerpts provided below.
Do not use outside knowledge or make claims beyond what the sources support.

Rules:
- Base every claim on the provided excerpts
- Cite sources inline using: [Source: filename, Category: material_category]
- If the excerpts don't contain enough to answer, say so clearly — do not guess
- Be specific and technical where the sources support it
- When comparing materials, structure your answer clearly"""

# ── Clients (initialized once at import time) ───────────────────────────────

openai_client = OpenAI()

chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection(name=COLLECTION_NAME)
# get_collection (not get_or_create) — fails loudly if collection doesn't exist.
# Run embed.py first if you see a collection not found error.

# ── Core functions ──────────────────────────────────────────────────────────

def _detect_materials(question: str) -> list[str]:
    """Return the list of material categories named in the question.

    Scans the question for known material aliases (longest match first to avoid
    'wool' matching before 'sheep wool'). Returns unique category names in the
    order they were found.
    """
    q = question.lower()
    found: list[str] = []
    # Sort aliases longest-first so multi-word terms match before substrings.
    for alias in sorted(MATERIAL_ALIASES, key=len, reverse=True):
        category = MATERIAL_ALIASES[alias]
        if alias in q and category not in found:
            found.append(category)
    return found


def _is_comparison_query(question: str) -> bool:
    """Detect whether the question is comparing two or more materials."""
    keywords = ("compare", "vs", "versus", "difference between", "better than",
                 "which is better", "pros and cons")
    q = question.lower()
    return any(kw in q for kw in keywords) or len(_detect_materials(question)) >= 2


def _extract_author(filename: str) -> str:
    """Extract the author key from a filename like 'Cork_Almusaed.pdf'.

    All source files follow the pattern [Topic]_[Author].pdf. The author
    portion is used as the deduplication key so that the same reference book
    filed under multiple material categories (e.g. Cob_Almusaed.pdf,
    Wheat straw_Almusaed.pdf, Natural fibers_Almusaed.pdf) is treated as one
    source and capped together.

    Falls back to the full filename if the pattern doesn't match.
    """
    stem = filename.replace(".pdf", "")
    if "_" in stem:
        return stem.split("_", 1)[-1].strip().lower()
    return stem.lower()


def deduplicate(chunks: list[dict], max_per_source: int = MAX_CHUNKS_PER_AUTHOR) -> list[dict]:
    """Keep only the top-ranked chunk(s) from each unique author.

    Chunks arrive sorted by similarity (best first). We walk the list and
    track how many chunks we've kept from each author. Once an author hits
    the cap, further chunks from any of their files are dropped — even if
    filed under different material categories.

    This prevents a single comprehensive reference (Almusaed, Hanaor, Chen)
    from filling all slots across its many topic-labeled copies.
    """
    seen: dict[str, int] = {}
    result = []
    for chunk in chunks:
        author = _extract_author(chunk["metadata"]["source"])
        count = seen.get(author, 0)
        if count < max_per_source:
            result.append(chunk)
            seen[author] = count + 1
    return result


# ── MAX_CHUNKS_PER_SOURCE renamed to MAX_CHUNKS_PER_AUTHOR ─────────────────
MAX_CHUNKS_PER_SOURCE = MAX_CHUNKS_PER_AUTHOR  # backwards-compat alias


def _fetch(question: str, k: int, material_filter: str | None = None) -> list[dict]:
    """Single ChromaDB query. Returns raw chunks sorted by similarity.

    Separated from retrieve() so comparison queries can call it multiple
    times (once per material) and merge the results.
    """
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=[question],
    )
    query_vector = response.data[0].embedding

    where = {"material_category": material_filter} if material_filter else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "similarity": round(1 - dist, 4),
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite).
            # 1 - distance converts to similarity (1.0 = perfect match).
        })
    return chunks


def retrieve(question: str, k: int | None = None, material_filter: str | None = None) -> list[dict]:
    """Embed the question and fetch the most relevant chunks from ChromaDB.

    For single-material or general queries: one unfiltered fetch + deduplicate.

    For comparison queries (two or more materials detected): fetches separately
    for each named material so both are guaranteed representation, then merges
    with a general fetch and deduplicates by author.

    k: override chunk count per fetch if needed.
    material_filter: force a single category filter (bypasses comparison logic).
    """
    if material_filter:
        # Explicit filter — bypass comparison logic entirely.
        chunks = _fetch(question, k=k or TOP_K, material_filter=material_filter)
        return deduplicate(chunks)

    detected = _detect_materials(question)

    if len(detected) >= 2:
        # Comparison query: fetch per-material + general, then merge.
        per_k = k or TOP_K_PER_MATERIAL
        all_chunks: list[dict] = []
        for material in detected:
            all_chunks.extend(_fetch(question, k=per_k, material_filter=material))
        # General fetch catches cross-material papers that mention both.
        all_chunks.extend(_fetch(question, k=TOP_K_GENERAL))
        # Sort merged pool by similarity before deduplication.
        all_chunks.sort(key=lambda c: c["similarity"], reverse=True)
        return deduplicate(all_chunks)

    # Single-material or general query.
    chunks = _fetch(question, k=k or TOP_K)
    return deduplicate(chunks)


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a labelled context block for GPT-4o.

    Each excerpt is numbered and tagged with its source and similarity score
    so GPT-4o can cite them accurately in its answer.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk["metadata"]["source"]
        category = chunk["metadata"]["material_category"]
        score = chunk["similarity"]
        parts.append(
            f"[Excerpt {i} | {category} | {source} | similarity: {score}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def generate_answer(question: str, context: str) -> str:
    """Send the question and retrieved context to GPT-4o. Return the answer."""
    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Source excerpts:\n\n{context}\n\nQuestion: {question}",
            },
        ],
        temperature=0,
        # temperature=0 — fully deterministic. RAG answers should be consistent
        # and grounded, not creative. Raise only if responses feel too rigid.
    )
    return response.choices[0].message.content


def query(question: str, k: int = TOP_K, material_filter: str | None = None) -> dict:
    """Full pipeline: retrieve → build context → generate answer.

    This is the function to call from the app layer.

    Returns:
        {
            "answer": str,
            "sources": [
                {
                    "source": filename,
                    "material_category": category,
                    "similarity": float,
                    "text": chunk text
                },
                ...
            ]
        }
    """
    chunks = retrieve(question, k=k, material_filter=material_filter)
    context = build_context(chunks)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": [
            {
                "source": c["metadata"]["source"],
                "material_category": c["metadata"]["material_category"],
                "similarity": c["similarity"],
                "text": c["text"],
            }
            for c in chunks
        ],
    }


# ── Interactive CLI ─────────────────────────────────────────────────────────

def main():
    # Force UTF-8 output so Greek letters and special chars in academic text
    # don't crash the Windows terminal (which defaults to cp1252).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set.")
        return

    print("Eco Atlas — Sustainable Materials Knowledge Base")
    print(f"Collection: {COLLECTION_NAME} ({collection.count()} chunks)")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("Question: ").strip()
        if not question or question.lower() in ("quit", "exit"):
            break

        print("\nRetrieving...\n")
        result = query(question)

        print("Answer:")
        print(result["answer"])

        print("\nSources used:")
        for s in result["sources"]:
            print(f"  [{s['similarity']}]  {s['material_category']}  /  {s['source']}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
