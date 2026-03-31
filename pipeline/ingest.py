"""
ingest.py — Load and chunk PDFs from data/raw/

Walks data/raw/[Material Category]/[file].pdf
Extracts full document text with PyMuPDF, splits into semantically coherent
chunks using OpenAI embeddings to find natural topic boundaries, saves to
data/processed/

Output: one JSON file per PDF in data/processed/
Each file contains a list of chunks with text + metadata.

Run: python pipeline/ingest.py
"""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# ── Chunking config ────────────────────────────────────────────────────────

# How SemanticChunker decides where a topic boundary is:
#   "percentile"         — split where similarity drops below the Nth percentile
#                          across all sentence pairs in the document
#   "standard_deviation" — split where similarity drops more than N std devs
#                          below the mean (more aggressive on varied docs)
#   "interquartile"      — uses IQR of similarity scores; robust to outliers
BREAKPOINT_THRESHOLD_TYPE = "percentile"

# The threshold value for the method above.
# For "percentile": 95 = only split at the bottom 5% of similarity scores.
# Higher → fewer, larger chunks. Lower → more, smaller chunks.
# 95 is conservative — good for dense academic text where topics shift slowly.
BREAKPOINT_THRESHOLD_AMOUNT = 95

# Drop any chunk shorter than this — catches noise, headers, page numbers,
# and reference list fragments that aren't useful for retrieval.
MIN_CHUNK_CHARS = 150

# Hard-split any chunk longer than this to stay within embedding token limits.
# SemanticChunker rarely produces chunks this large, but this is a safety net.
MAX_CHUNK_CHARS = 3000

# ── Embeddings + splitter (initialized once, reused across all documents) ──

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # text-embedding-3-small: fast, cheap, strong semantic signal for RAG.
    # 1536 dimensions by default — enough resolution for material science topics.
)

splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type=BREAKPOINT_THRESHOLD_TYPE,
    breakpoint_threshold_amount=BREAKPOINT_THRESHOLD_AMOUNT,
    # SemanticChunker works by: splitting text into sentences → embedding each
    # sentence → measuring cosine similarity between adjacent sentences →
    # inserting a chunk boundary wherever similarity drops past the threshold.
)

# ── Core functions ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF as a single string.
    Pages are joined with double newlines to preserve section breaks
    without confusing the sentence splitter.
    """
    doc = fitz.open(pdf_path)
    pages = [page.get_text().strip() for page in doc if page.get_text().strip()]
    doc.close()
    return "\n\n".join(pages)


def guard_chunk_size(chunks: list[str]) -> list[str]:
    """Apply min/max size guards after semantic splitting.

    - Below MIN_CHUNK_CHARS: discard (noise, stray headers, lone references)
    - Above MAX_CHUNK_CHARS: hard-split at the limit with no overlap
      This is a fallback — the semantic splitter handles most cases correctly.
    """
    guarded = []
    for chunk in chunks:
        if len(chunk) < MIN_CHUNK_CHARS:
            continue
        if len(chunk) > MAX_CHUNK_CHARS:
            for i in range(0, len(chunk), MAX_CHUNK_CHARS):
                sub = chunk[i:i + MAX_CHUNK_CHARS].strip()
                if len(sub) >= MIN_CHUNK_CHARS:
                    guarded.append(sub)
        else:
            guarded.append(chunk)
    return guarded


def chunk_document(full_text: str, metadata: dict) -> list[dict]:
    """Semantically chunk a full document. Attaches metadata to each chunk."""
    raw_chunks = splitter.split_text(full_text)
    clean_chunks = guard_chunk_size(raw_chunks)

    return [
        {
            "text": chunk,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "char_count": len(chunk),
            }
        }
        for i, chunk in enumerate(clean_chunks)
    ]


def process_pdf(pdf_path: Path) -> list[dict]:
    """Full pipeline for a single PDF: extract → chunk → return chunks."""
    metadata = {
        "source": pdf_path.name,
        "material_category": pdf_path.parent.name,
        "source_path": str(pdf_path),
    }

    full_text = extract_text_from_pdf(pdf_path)
    if not full_text.strip():
        print(f"  [!] No text extracted: {pdf_path.name}")
        return []

    return chunk_document(full_text, metadata)


def md5(path: Path) -> str:
    """Return MD5 hash of a file. Used to detect duplicate PDFs."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def group_by_hash(pdf_files: list[Path]) -> dict[str, list[Path]]:
    """Group PDF paths by file hash. Duplicates share the same key."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in pdf_files:
        groups[md5(path)].append(path)
    return groups


def save_chunks(chunks: list[dict], output_path: Path):
    """Save chunks to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


def retag_chunks(chunks: list[dict], pdf_path: Path) -> list[dict]:
    """Return a copy of chunks with metadata retagged for a different PDF path.
    Used when the same document appears in multiple material category folders.
    """
    return [
        {
            "text": c["text"],
            "metadata": {
                **c["metadata"],
                "source": pdf_path.name,
                "material_category": pdf_path.parent.name,
                "source_path": str(pdf_path),
            }
        }
        for c in chunks
    ]


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add your key to the .env file.")
        return

    pdf_files = sorted(RAW_DIR.rglob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in data/raw/")
        return

    print(f"Found {len(pdf_files)} PDFs — scanning for duplicates...\n")

    groups = group_by_hash(pdf_files)
    dupes = {h: paths for h, paths in groups.items() if len(paths) > 1}

    if dupes:
        for paths in dupes.values():
            categories = [p.parent.name for p in paths]
            print(f"  [dup] Duplicate detected ({len(paths)} copies): {paths[0].name}")
            print(f"    -> Will process once, tag across: {', '.join(categories)}\n")

    total_chunks = 0
    skipped = 0

    for file_hash, paths in groups.items():
        is_duplicate = len(paths) > 1

        # Check if all output files already exist — skip the whole group if so
        output_paths = [
            PROCESSED_DIR / p.relative_to(RAW_DIR).with_suffix(".json")
            for p in paths
        ]
        if all(op.exists() for op in output_paths):
            for p in paths:
                print(f"  [skip] Already processed: {p.name}")
                skipped += 1
            continue

        # Use the first path as the canonical source for extraction + chunking
        canonical = paths[0]
        print(f"  Processing: {canonical.parent.name} / {canonical.name}")
        if is_duplicate:
            print(f"    (shared with {len(paths) - 1} other folder(s))")

        chunks = process_pdf(canonical)
        if not chunks:
            continue

        # Save one JSON per path, each retagged with its own material_category
        for pdf_path, output_path in zip(paths, output_paths):
            if output_path.exists():
                print(f"    [skip] Already exists: {output_path}")
                skipped += 1
                continue

            tagged = retag_chunks(chunks, pdf_path) if pdf_path != canonical else chunks
            save_chunks(tagged, output_path)
            print(f"    -> {len(tagged)} chunks -> {output_path.parent.name}/{output_path.name}")

        total_chunks += len(chunks) * len(paths)

    print(f"\nDone. {total_chunks} total chunks across {len(pdf_files) - skipped} output files.")
    if skipped:
        print(f"{skipped} files skipped (already processed).")


if __name__ == "__main__":
    main()
