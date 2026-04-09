"""
chat.py — EcoAtlas demo chatbot

Streamlit UI for the RAG pipeline. Prototype build — RAG mechanics visible.

Run: streamlit run app/chat.py
"""

import sys
from pathlib import Path

# Allow imports from project root (pipeline/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from pipeline.query import query, _detect_materials, COLLECTION_NAME, collection

# ── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Eco Atlas — Materials Assistant",
    page_icon="🌿",
    layout="wide",
)

# ── Helper ───────────────────────────────────────────────────────────────────

def render_web_results(web_results: list[dict]) -> None:
    """Render web sources as a reference list below the answer.

    Content from these sources is already incorporated into the answer with
    [Web: url] inline citations — this panel just makes the links easy to open.
    """
    if not web_results:
        return
    st.divider()
    st.markdown("#### 🌐 Web Sources")
    st.caption("These pages were searched and cited in the answer above where marked [Web: url].")
    for r in web_results:
        st.markdown(f"- **[{r['title']}]({r['url']})**")


def render_sources(sources: list[dict], detected_materials: list[str]) -> None:
    """Render the RAG mechanics panel below an assistant message."""
    st.divider()

    if detected_materials:
        st.caption(
            "**Materials detected in query:** "
            + " · ".join(f"`{m}`" for m in detected_materials)
            + "  —  per-material retrieval used"
        )

    st.caption(f"**{len(sources)} source chunks retrieved**")

    cols = st.columns([3, 3, 1])
    cols[0].caption("**File**")
    cols[1].caption("**Category**")
    cols[2].caption("**Similarity**")

    for s in sources:
        cols = st.columns([3, 3, 1])
        cols[0].caption(s["source"])
        cols[1].caption(s["material_category"])
        score = s["similarity"]
        color = "green" if score >= 0.65 else "orange" if score >= 0.55 else "red"
        cols[2].markdown(f":{color}[**{score}**]")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🌿 Eco Atlas")
    st.caption("Sustainable Building Materials Knowledge Base")
    st.divider()

    st.subheader("Knowledge Base")
    st.metric("Total chunks", f"{collection.count():,}")
    st.metric("Collection", COLLECTION_NAME)
    st.caption("Source: 55 academic PDFs · 16 material categories")

    st.divider()

    st.subheader("Materials covered")
    for m in [
        "Bamboo", "Bio Clay Plaster", "Cellulose", "Cob",
        "Compacted Soil", "Concrete variants", "Cork", "Hemp",
        "Linoleum", "Mycelium", "Natural Plant Fibers",
        "PU Foam (plant-based)", "Recycled Plastic + Rubber",
        "Sheep Wool", "Solar Shingles", "Wheat Straw / Straw Bales",
    ]:
        st.caption(f"• {m}")

    st.divider()

    st.subheader("Response mode")
    mode = st.radio(
        "How should the assistant respond?",
        options=["factual", "speculative"],
        format_func=lambda x: "Factual" if x == "factual" else "Speculative",
        index=0,
        help=(
            "**Factual** — grounded answers drawn directly from academic research.\n\n"
            "**Speculative** — creative, design-led responses imagining how materials could look and feel in use."
        ),
    )
    if mode == "speculative":
        st.caption("Speculative mode: answers go beyond the sources to imagine real-world use.")

    st.divider()
    st.caption("OpenAI text-embedding-3-small → ChromaDB → GPT-4o")
    st.caption("Semantic chunking · per-material fetch for comparisons")

    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────

st.title("Sustainable Materials Assistant")
if mode == "speculative":
    st.caption(
        "Speculative mode — answers imagine how materials could look, feel, and perform in real spaces. "
        "Sources shown below each response."
    )
else:
    st.caption(
        "Ask anything about sustainable building materials. "
        "All answers are grounded in academic research — sources shown below each response."
    )

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            st.markdown("#### 📚 From the Knowledge Base")
            render_sources(msg["sources"], msg.get("detected_materials", []))
            render_web_results(msg.get("web_results", []))

# ── Input ─────────────────────────────────────────────────────────────────────

if prompt := st.chat_input("Ask about sustainable building materials..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    detected = _detect_materials(prompt)

    history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg["role"] in ("user", "assistant")
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from knowledge base..."):
            result = query(prompt, history=history, mode=mode)
        st.markdown("#### 📚 From the Knowledge Base")
        st.markdown(result["answer"])
        render_sources(result["sources"], detected)
        render_web_results(result.get("web_results", []))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "detected_materials": detected,
        "web_results": result.get("web_results", []),
        "mode": mode,
    })
