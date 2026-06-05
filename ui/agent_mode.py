"""
ui/agent_mode.py
─────────────────
Agent tab: one-click document operations using RAG-retrieved context.

Operations:
  - Summarize       → structured bullet-point summary via RAG
  - Key Topics      → topic extraction with page citations
  - Extract Tables  → table/structured data extraction
  - Generate Notes  → study-style condensed notes

All operations use RAG retrieval (not full document dump) when the document
is large, falling back to full text for small documents (<20 chunks).
"""

from __future__ import annotations

import streamlit as st

from core.rag_engine import RAGEngine


def render_agent_mode() -> None:
    docs: dict = st.session_state.docs
    rag: RAGEngine = st.session_state.rag

    st.subheader("🤖 Document Agent")
    st.caption("One-click intelligent document operations.")

    if not docs:
        st.info("Upload at least one PDF to use the Agent.")
        return

    doc_choice = st.selectbox("Choose document", list(docs.keys()))
    doc = docs[doc_choice]

    col1, col2, col3, col4 = st.columns(4)
    action = None
    if col1.button("📝 Summarize",    use_container_width=True): action = "summarize"
    if col2.button("🔍 Key Topics",   use_container_width=True): action = "topics"
    if col3.button("📊 Extract Data", use_container_width=True): action = "tables"
    if col4.button("🗒 Study Notes",  use_container_width=True): action = "notes"

    if action:
        with st.spinner("Agent working…"):
            try:
                result = _run_agent_action(action, doc_choice, doc, rag)
                _render_result(action, result, doc_choice)
            except Exception as exc:
                st.error(f"Agent error: {exc}")

    with st.expander("📖 View raw document text (first 2 000 chars)"):
        st.text(doc.full_text[:2000])


# ── Action dispatchers ────────────────────────────────────────────────────────

_PROMPTS = {
    "summarize": (
        "You are a document analyst. Produce a structured summary of the following "
        "document content in clear bullet points grouped by topic. "
        "Include the most important facts and any key conclusions.\n\n"
    ),
    "topics": (
        "List and explain the main topics covered in the following document content. "
        "For each topic, write 2-3 sentences explaining what the document says about it.\n\n"
    ),
    "tables": (
        "Extract any tables or structured/tabular data present in the following content. "
        "Format each table as a Markdown table. "
        "If no explicit tables exist, identify and format any structured data (lists of items "
        "with multiple attributes, numeric data, comparisons) as a Markdown table. "
        "Note any table that appears to be truncated.\n\n"
    ),
    "notes": (
        "Generate concise, well-organised study notes from the following document content. "
        "Use headings (##) for major sections and bullet points for key facts. "
        "Emphasise definitions, important numbers, and conclusions.\n\n"
    ),
}


def _run_agent_action(action: str, filename: str, doc, rag: RAGEngine) -> str:
    """
    For agent actions we use RAG to pull representative chunks from the
    document rather than dumping the full text.  This scales to large PDFs.

    We retrieve up to 10 chunks (double the chat default) to give the LLM
    a broad view of the document.
    """
    # Build a broad retrieval query for each action type
    retrieval_queries = {
        "summarize": "main topics summary overview introduction conclusion",
        "topics":    "key themes topics subjects discussed",
        "tables":    "table data figures statistics numbers comparison",
        "notes":     "important facts definitions key points conclusions",
    }
    query = retrieval_queries[action]

    chunks = rag.vector_store.query(
        query, n_results=10, filter_sources=[filename]
    )

    if chunks:
        context = "\n\n".join(
            f"[Page {c.page}] {c.text}" for c in chunks
        )
    else:
        # Fallback: use raw full text truncated
        context = doc.full_text[:40_000]

    prompt = _PROMPTS[action] + f"Document: {filename}\n\n{context}"
    return rag.simple_ask(prompt)


def _render_result(action: str, result: str, doc_name: str) -> None:
    labels = {
        "summarize": "📝 Summary",
        "topics":    "🔍 Key Topics",
        "tables":    "📊 Extracted Data",
        "notes":     "🗒 Study Notes",
    }
    st.subheader(labels[action])
    st.markdown(result)
    st.download_button(
        f"⬇ Download {labels[action]}",
        result,
        file_name=f"{action}_{doc_name}.txt",
        use_container_width=False,
    )
