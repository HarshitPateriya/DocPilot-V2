"""
ui/chat_mode.py
────────────────
Chat tab: RAG-powered Q&A with deterministic citations and collapsible sources.

Every answer shows:
  1. LLM prose response.
  2. Sources block: "**Sources:** [file.pdf · p.N], ..."
  3. Collapsible expander with each retrieved chunk's full text + metadata.

The citations and sources come ENTIRELY from chunk metadata — the LLM does
not generate any citation text.
"""

from __future__ import annotations

import streamlit as st

from core.rag_engine import RAGAnswer, RAGEngine


def render_chat_mode() -> None:
    """Render the full Chat tab UI."""
    st.subheader("💬 Chat with your documents")
    st.caption(
        "Answers are grounded in retrieved document chunks. "
        "Citations and sources are generated from metadata, not by the LLM."
    )

    docs: dict = st.session_state.docs
    rag: RAGEngine = st.session_state.rag

    if not docs:
        st.info("Upload at least one PDF to start chatting.")
        return

    # ── Source filter (optional) ──────────────────────────────────────────────
    all_sources = list(docs.keys())
    if len(all_sources) > 1:
        selected_sources = st.multiselect(
            "Filter to specific documents (leave blank = all)",
            options=all_sources,
            default=[],
            help="Restricts retrieval to chosen documents only.",
        )
        filter_sources = selected_sources if selected_sources else None
    else:
        filter_sources = None

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        _render_message(msg)

    # ── Input ─────────────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about your documents…")

    if question:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("Retrieving and generating answer…"):
            result: RAGAnswer = rag.answer(
                question=question,
                filter_sources=filter_sources,
            )

        # Persist to history
        st.session_state.messages.append({
            "question":     question,
            "answer":       result.answer,
            "citations":    result.citations,
            "source_chunks": [
                {
                    "text":     c.text,
                    "source":   c.source,
                    "page":     c.page,
                    "score":    c.relevance_score,
                }
                for c in result.source_chunks
            ],
        })
        st.rerun()


# ── Private helpers ───────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single persisted chat turn (question + answer + sources)."""
    with st.chat_message("user"):
        st.write(msg["question"])

    with st.chat_message("assistant"):
        st.write(msg["answer"])

        # ── Deterministic citations block ─────────────────────────────────────
        if msg.get("citations"):
            st.markdown(
                "**Sources:** " + " · ".join(msg["citations"])
            )

        # ── Collapsible source chunks ─────────────────────────────────────────
        chunks = msg.get("source_chunks", [])
        if chunks:
            with st.expander(f"📄 View {len(chunks)} retrieved source chunk(s)"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(
                        f"**Chunk {i}** — `{chunk['source']}`, "
                        f"Page {chunk['page']} "
                        f"_(relevance: {chunk['score']:.0%})_"
                    )
                    st.text(chunk["text"][:600] + ("…" if len(chunk["text"]) > 600 else ""))
                    if i < len(chunks):
                        st.divider()
