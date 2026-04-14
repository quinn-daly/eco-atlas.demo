"""
api/main.py — EcoAtlas REST API

FastAPI wrapper around the RAG pipeline for production deployment.
Serves the chat widget at / and exposes the pipeline at POST /chat.

Run locally:
    uvicorn api.main:app --reload --port 8000

Deploy:
    Railway / Render — see Dockerfile at project root.
"""

import logging
import os
import sys
from pathlib import Path

# Allow imports from project root (pipeline/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Startup guard ─────────────────────────────────────────────────────────────
# Fail fast before the pipeline initialises so Railway surfaces the error in
# build logs rather than on the first user request.
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY is not set. "
        "Add it as an environment variable in Railway before deploying."
    )

from pipeline.query import query

app = FastAPI(title="EcoAtlas API", docs_url=None, redoc_url=None)

# CORS: Wix makes requests from its own CDN origin — allow all for iframe embeds.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WIDGET_PATH = Path(__file__).parent / "widget.html"


class Message(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []
    mode: str = "factual"   # "factual" | "speculative"


@app.get("/")
def serve_widget():
    """Serve the embeddable chat widget."""
    return FileResponse(WIDGET_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Run the full RAG pipeline and return an answer with sources.

    Returns:
        {
            "answer": str,
            "sources": [{"source": str, "material_category": str, "similarity": float}],
            "web_results": [{"title": str, "url": str, "snippet": str}]
        }
    """
    try:
        history = [{"role": m.role, "content": m.content} for m in req.history]
        result = query(req.message, history=history, mode=req.mode)
        return result
    except Exception as exc:
        logger.exception("Pipeline error on message: %r", req.message)
        # Return a JSON body the widget can display rather than an HTML 500 page.
        return JSONResponse(
            status_code=500,
            content={"error": "The assistant encountered an error. Please try again."},
        )
