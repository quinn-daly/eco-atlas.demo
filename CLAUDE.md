# CLAUDE.md — Eco Atlas

## Read first
See D:/_context/about-me.md for working style and global preferences.

## What this is
A web platform for students researching sustainable building materials.
Users interact with an AI chatbot to compare materials, plan builds,
and generate images — rather than clicking through static content.
This is a consulting project. Client owns the Wix Studio site.
My role is building and integrating the AI backend.

## Current status
Active development. Source documents ingested and folder structure in place.
Pipeline build in progress — next step is ingest.py and chunking strategy.

## Architecture Overview
Three distinct layers — keep them mentally separate at all times:

**Layer 1 — Frontend (Wix Studio)**
- Client-owned Wix Studio site
- Custom chatbot UI embedded via Wix Velo
- Sends user queries to backend API
- Displays responses + sources returned from backend
- We do NOT control this layer deeply — Wix is the constraint

**Layer 2 — Backend API**
- External service Wix Velo calls
- Handles RAG logic: embed query → retrieve chunks → generate response
- Returns answer + source references to Wix
- Stack TBD: leaning OpenAI for LLM + embeddings
- Vector database TBD — decide based on scale and budget

**Layer 3 — Data / Knowledge Base**
- Source: existing sustainable materials research document
- Needs to be chunked, embedded, and stored in vector DB
- This is the core intelligence of the system

## Key Features (planned)
- Material comparison via natural language queries
- Build planning assistance
- Image generation for material visualization
- Source-cited responses (RAG grounded, not hallucinated)
- Internet access for up-to-date material information

## Domain Vocabulary
- **RAG**: Retrieval Augmented Generation — grounding LLM responses
  in retrieved document chunks rather than model memory
- **Chunk**: a segment of the source document stored as a vector
- **Embedding**: numerical representation of text for similarity search
- **Velo**: Wix's backend JavaScript environment for API calls
- **Grounded response**: answer derived from retrieved sources,
  not model hallucination

## Stack (decided)
- Frontend: Wix Studio + Velo (client constraint, not negotiable)
- LLM: OpenAI gpt-4o
- Embeddings: OpenAI text-embedding-3-small
- Vector DB: ChromaDB (local, for pipeline development — may swap for hosted later)
- Framework: LangChain + langchain-community
- Backend hosting: TBD

## Dependencies
```bash
pip install openai pymupdf chromadb python-dotenv langchain langchain-community langchain-openai langchain-experimental
```

## Decisions Still to Make
- [ ] Backend hosting (where does the API live?)
- [ ] Chunking strategy for source documents
- [ ] Image generation model (DALL-E vs Stable Diffusion vs other)
- [ ] How sources are cited and displayed in UI
- [ ] Whether to swap ChromaDB for hosted vector DB (Pinecone / Supabase pgvector)

## Current Priorities
- Build pipeline/ingest.py — load and chunk PDFs from data/raw/
- Build pipeline/embed.py — generate and store embeddings in ChromaDB
- Build pipeline/query.py — retrieval and response logic
- Test pipeline end-to-end before touching Wix integration

## Out of scope (for now)
- User accounts or saved sessions
- Mobile app
- Real-time collaboration
- Any changes to Wix site structure — UI is client's domain

## Key Constraint
Wix Studio and Velo are the frontend — this is non-negotiable.
All AI complexity lives outside Wix. Keep the Velo code simple:
receive query → call API → display response. Nothing more.

## Source Data
55 research PDFs organized by material category in `data/raw/`:
- Bamboo, Bio Clay Plaster, Cellulose, Cob, Compacted Soil
- Concrete (Hempcrete, Ashcrete, Timbercrete, Ferrock)
- Cork, Hemp, Linoleum, Mycelium, Natural Plant Fibers
- Plant-based Rigid PU Foam, Recycled Plastic + Rubber
- Sheep Wool, Solar Shingles, Wheat Straw / Straw Bales

Each folder contains 1–7 PDFs named `[Topic]_[Author].pdf`.
Processed/chunked versions will live in `data/processed/`.
