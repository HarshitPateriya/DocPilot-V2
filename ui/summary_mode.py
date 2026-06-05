"""
ui/summary_mode.py
───────────────────
Summary tab: generates document summaries via direct LLM call.

Note: Summaries use the full document text (not RAG retrieval) because
summaries by nature need the whole document.  We truncate to 60k chars
to stay within Groq's input limits.

For multi-document summaries we use the first 15k chars per document
(5 docs × 15k = 75k) which covers most documents adequately.
"""

from __future__ import annotations

import streamlit as st

from core.document_processor import ProcessedDocument
from core.rag_engine import RAGEngine


MAX_CHARS_PER_DOC  = 15_000   # per-doc limit for multi-doc summary
MAX_CHARS_SINGLE   = 60_000   # limit for single-doc summary


def render_summary_mode() -> None:
    docs: dict = st.session_state.docs
    rag: RAGEngine = st.session_state.rag

    st.subheader("📋 Document Summaries")

    if not docs:
        st.info("Upload at least one PDF to generate a summary.")
        return

    doc_options = list(docs.keys()) + (["All documents"] if len(docs) > 1 else [])
    choice = st.selectbox("Choose document to summarise", doc_options)

    style = st.selectbox(
        "Summary style",
        ["Concise (bullet points)", "Detailed (prose)", "Executive (3 sentences)"],
    )
    style_instruction = {
        "Concise (bullet points)": "Summarise in clear bullet points grouped by topic.",
        "Detailed (prose)":        "Write a detailed prose summary with section headings.",
        "Executive (3 sentences)": "Write exactly 3 sentences: what the document is, its key finding, and its conclusion.",
    }[style]

    if st.button("Generate Summary", type="primary"):
        with st.spinner("Generating summary…"):
            try:
                if choice == "All documents":
                    summary = _summarise_all(docs, rag, style_instruction)
                else:
                    summary = _summarise_one(docs[choice], rag, style_instruction)

                st.subheader("Summary")
                st.markdown(summary)
                st.download_button(
                    "⬇ Download Summary",
                    summary,
                    file_name=f"summary_{choice.replace(' ', '_')}.txt",
                )
            except Exception as exc:
                st.error(f"Error generating summary: {exc}")


def _summarise_one(doc: ProcessedDocument, rag: RAGEngine, style: str) -> str:
    text = doc.full_text[:MAX_CHARS_SINGLE]
    prompt = (
        f"{style}\n\n"
        f"Document: {doc.filename}\n"
        f"Pages: {doc.num_pages}\n\n"
        f"{text}"
    )
    return rag.simple_ask(prompt)


def _summarise_all(docs: dict, rag: RAGEngine, style: str) -> str:
    parts = []
    for fname, doc in docs.items():
        excerpt = doc.full_text[:MAX_CHARS_PER_DOC]
        parts.append(f"=== {fname} ===\n{excerpt}")

    combined = "\n\n".join(parts)
    prompt = (
        f"{style}\n"
        f"You are summarising {len(docs)} documents. "
        f"Note topics each document covers and any overlap.\n\n"
        f"{combined}"
    )
    return rag.simple_ask(prompt)
