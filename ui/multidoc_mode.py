"""
ui/multidoc_mode.py
────────────────────
Multi-Document Analysis tab.

Three preset analyses + a freeform cross-document question:
  1. Compare Documents  — side-by-side topic comparison
  2. Find Contradictions — conflicting claims or data
  3. Common Topics      — themes present across all docs

For each analysis we retrieve chunks from ALL documents (no source filter)
and send them together.  The system prompt instructs the LLM to label
each claim with the document it came from.

Deterministic citations: retrieved chunk metadata → citation footer,
same as Chat mode.
"""

from __future__ import annotations

import streamlit as st

from core.rag_engine import RAGEngine


def render_multidoc_mode() -> None:
    docs: dict = st.session_state.docs
    rag: RAGEngine = st.session_state.rag

    st.subheader("🔍 Multi-Document Intelligence")

    if len(docs) < 2:
        st.warning("Upload at least **2 PDFs** to use Multi-Document Analysis.")
        return

    doc_names = list(docs.keys())
    st.markdown(f"**Loaded:** {', '.join(f'`{n}`' for n in doc_names)}")
    st.divider()

    col1, col2, col3 = st.columns(3)
    action = None
    if col1.button("⚖️ Compare Documents",  use_container_width=True): action = "compare"
    if col2.button("⚡ Find Contradictions", use_container_width=True): action = "contradict"
    if col3.button("🔗 Common Topics",       use_container_width=True): action = "common"

    if action:
        with st.spinner("Analysing across all documents…"):
            try:
                result, chunks = _run_analysis(action, doc_names, rag)
                _render_analysis_result(result, chunks)
            except Exception as exc:
                st.error(f"Analysis error: {exc}")

    st.divider()
    st.markdown("**Custom cross-document question**")
    custom_q = st.text_input(
        "Ask something across all documents…",
        placeholder="e.g. Which document has the highest revenue projection?",
    )
    if custom_q and st.button("Ask", key="multidoc_ask"):
        with st.spinner("Retrieving and answering…"):
            try:
                rag_result = rag.answer(question=custom_q)  # all sources
                st.markdown(rag_result.answer)
                if rag_result.citations:
                    st.markdown("**Sources:** " + " · ".join(rag_result.citations))
                _show_source_chunks(rag_result.source_chunks)
            except Exception as exc:
                st.error(f"Error: {exc}")


# ── Analysis helpers ──────────────────────────────────────────────────────────

_ANALYSIS_QUERIES = {
    "compare":     "main topics overview introduction purpose",
    "contradict":  "claims data statistics findings conclusions",
    "common":      "shared themes recurring concepts mentioned topics",
}

_ANALYSIS_PROMPTS = {
    "compare": (
        "You are comparing multiple documents. For each major topic area, explain "
        "what each document says about it. Use a structured format with '### Topic: X' "
        "headings and a sub-section per document. Be specific and factual.\n\n"
        "Documents analysed: {names}\n\n{context}"
    ),
    "contradict": (
        "Identify any contradictions, inconsistencies, or conflicting information "
        "across the following document chunks. For each contradiction found:\n"
        "- State the claim from Document A\n"
        "- State the conflicting claim from Document B\n"
        "- Explain why they conflict\n"
        "If no contradictions are found, state that clearly.\n\n"
        "Documents: {names}\n\n{context}"
    ),
    "common": (
        "Identify topics, themes, facts, or concepts that appear across "
        "MULTIPLE documents from this set: {names}.\n"
        "For each shared topic:\n"
        "- Name the topic\n"
        "- List which documents address it\n"
        "- Briefly describe each document's perspective\n\n"
        "{context}"
    ),
}


def _run_analysis(action: str, doc_names: list, rag: RAGEngine):
    """Retrieve chunks from all documents and run the requested analysis."""
    from core.vector_store import RetrievedChunk

    query = _ANALYSIS_QUERIES[action]
    # Retrieve from all sources (no filter)
    chunks = rag.vector_store.query(query, n_results=12)

    context = "\n\n".join(
        f"[{c.source}, Page {c.page}]\n{c.text}" for c in chunks
    )
    prompt = _ANALYSIS_PROMPTS[action].format(
        names=", ".join(doc_names),
        context=context,
    )
    result = rag.simple_ask(prompt, max_tokens=2000)
    return result, chunks


def _render_analysis_result(result: str, chunks) -> None:
    st.markdown(result)

    # Citations from metadata
    from core.rag_engine import _build_citations
    citations = _build_citations(chunks)
    if citations:
        st.markdown("**Sources:** " + " · ".join(citations))

    _show_source_chunks(chunks)
    st.download_button("⬇ Download Analysis", result, "analysis.txt")


def _show_source_chunks(chunks) -> None:
    if not chunks:
        return
    with st.expander(f"📄 View {len(chunks)} retrieved source chunk(s)"):
        for i, c in enumerate(chunks, 1):
            st.markdown(
                f"**Chunk {i}** — `{c.source}`, Page {c.page}"
            )
            text = c.text if isinstance(c.text, str) else c.get("text", "")
            st.text(text[:400] + ("…" if len(text) > 400 else ""))
            if i < len(chunks):
                st.divider()
