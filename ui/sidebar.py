"""
ui/sidebar.py
──────────────
Sidebar: file upload, processing pipeline trigger, document status panel.

This module owns ALL side-panel UI logic so that app.py stays clean.

Session state keys managed here:
    st.session_state.docs        → {filename: ProcessedDocument}
    st.session_state.rag         → RAGEngine instance
    st.session_state.active_mode → current tab string
"""

from __future__ import annotations

import streamlit as st

from core.document_processor import chunk_document, process_pdf
from core.rag_engine import RAGEngine


def render_sidebar() -> str:
    """
    Render the sidebar and return the currently selected mode string.

    Side effects:
        - Initialises st.session_state.rag if missing.
        - Processes newly uploaded PDFs into the vector store.
        - Updates st.session_state.docs.

    Returns:
        Selected mode label (e.g. "💬 Chat").
    """
    _ensure_session_state()

    with st.sidebar:
        st.markdown("## 📚 DocPilot")
        st.caption("Production RAG · Powered by Groq + Llama 3.3")
        st.divider()

        # ── File uploader ─────────────────────────────────────────────────────
        uploaded_files = st.file_uploader(
            "Upload PDFs (max 5)",
            type="pdf",
            accept_multiple_files=True,
            help="Scanned PDFs are automatically processed with OCR.",
        )

        if uploaded_files:
            _process_new_uploads(uploaded_files[:5])

        # ── Indexed documents panel ───────────────────────────────────────────
        if st.session_state.docs:
            st.divider()
            st.markdown("**Indexed Documents**")
            rag: RAGEngine = st.session_state.rag
            for fname, doc in st.session_state.docs.items():
                chunk_count = rag.vector_store.indexed_sources.get(fname, 0)
                ocr_info = (
                    f" · OCR: p.{','.join(map(str, doc.ocr_pages))}"
                    if doc.ocr_pages
                    else ""
                )
                st.markdown(
                    f"📄 `{fname}`  \n"
                    f"  {doc.num_pages} pages · {chunk_count} chunks{ocr_info}"
                )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑 Remove All", use_container_width=True):
                    rag.vector_store.reset()
                    st.session_state.docs = {}
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("🧹 Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    rag.clear_memory()
                    st.rerun()

        # ── Mode selector ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("**Mode**")
        mode = st.radio(
            "mode",
            options=[
                "💬 Chat",
                "📋 Summary",
                "🤖 Agent",
                "🧠 Study Mode",
                "🔍 Multi-Doc Analysis",
            ],
            label_visibility="collapsed",
        )

        # ── Memory status ─────────────────────────────────────────────────────
        if st.session_state.get("rag"):
            rag = st.session_state.rag
            turns = rag.memory.turn_count
            if turns:
                st.divider()
                st.caption(f"💬 Memory: {turns} turns in window")
                if rag.memory.summary:
                    with st.expander("View compressed summary"):
                        st.caption(rag.memory.summary)

        st.divider()
        st.caption("Built by Harshit · DocPilot v2")

    return mode


# ── Private helpers ───────────────────────────────────────────────────────────

def _ensure_session_state() -> None:
    """Initialise all required session state keys exactly once."""
    if "rag" not in st.session_state:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        if not api_key:
            st.error("GROQ_API_KEY not found in st.secrets.")
            st.stop()
        st.session_state.rag = RAGEngine(groq_api_key=api_key)

    for key, default in {
        "docs": {},
        "messages": [],
        "active_mode": "💬 Chat",
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _process_new_uploads(uploaded_files: list) -> None:
    """
    For each uploaded file not yet in session_state.docs:
      1. Extract text + OCR
      2. Chunk
      3. Embed + store in ChromaDB

    Uses st.spinner so the user sees progress.
    Files already indexed are skipped (idempotent).
    """
    rag: RAGEngine = st.session_state.rag
    new_files = [f for f in uploaded_files if f.name not in st.session_state.docs]

    for f in new_files:
        with st.spinner(f"Processing {f.name}…"):
            try:
                doc = process_pdf(f)
                chunks = chunk_document(doc)
                rag.vector_store.add_chunks(chunks)
                st.session_state.docs[f.name] = doc
                st.success(
                    f"✅ {f.name} — {doc.num_pages} pages, {len(chunks)} chunks"
                    + (f", OCR on {len(doc.ocr_pages)} page(s)" if doc.ocr_pages else "")
                )
            except Exception as exc:
                st.error(f"Failed to process {f.name}: {exc}")
