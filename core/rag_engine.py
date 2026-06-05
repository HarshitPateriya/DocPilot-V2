"""
core/rag_engine.py
───────────────────
The RAG (Retrieval-Augmented Generation) pipeline.

Flow for every user question:
  1. Query ChromaDB → top-K relevant chunks  (RetrievedChunk list)
  2. Build context block from chunk texts
  3. Inject context + conversation memory into the Groq prompt
  4. Call Groq API (Llama 3.3 70B)
  5. Attach citations programmatically from chunk metadata
  6. Return answer + citations + source chunks

KEY DESIGN: Citations come from metadata, NOT from the LLM.
  The LLM is instructed to produce clean prose without inline citation marks.
  After the answer is generated, we append a citations block built from the
  retrieved chunk metadata (filename + page number).  This makes citations
  100% deterministic and always accurate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from groq import Groq

from core.memory import ConversationMemory
from core.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
N_RETRIEVE = 5      # chunks retrieved per query
MAX_TOKENS = 1500   # answer token budget


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are DocPilot, a precise document assistant.

Rules:
1. Answer ONLY using information present in the RETRIEVED DOCUMENT CONTEXT provided below.
2. If the context does not contain enough information to answer, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
3. Write clear, well-structured prose. Use bullet points only when listing genuinely enumerable items.
4. Do NOT include citation markers like [1], (page 4), or footnotes in your answer.
   Citations will be added automatically from document metadata after your response.
5. Do NOT hallucinate facts, statistics, or quotes not present in the context.
6. When asked to compare documents, structure your answer with a clear section per document."""


# ── Answer model ──────────────────────────────────────────────────────────────

@dataclass
class RAGAnswer:
    """
    Complete result of a RAG query.

    Fields:
        answer:          The LLM-generated prose answer.
        citations:       Deduplicated, sorted list of "filename · p.N" strings.
        source_chunks:   Full RetrievedChunk list (for UI source display).
        retrieval_count: How many chunks were retrieved.
    """
    answer:          str
    citations:       List[str]         = field(default_factory=list)
    source_chunks:   List[RetrievedChunk] = field(default_factory=list)
    retrieval_count: int               = 0

    def formatted_answer(self) -> str:
        """Answer text with a citations block appended."""
        if not self.citations:
            return self.answer
        cite_block = "\n\n**Sources:**\n" + "\n".join(
            f"- {c}" for c in self.citations
        )
        return self.answer + cite_block


# ── RAGEngine ─────────────────────────────────────────────────────────────────

class RAGEngine:
    """
    Orchestrates retrieval + generation + citation assembly.

    One instance per Streamlit session; holds a VectorStore and a
    ConversationMemory.  Created once and stored in st.session_state.

    Args:
        groq_api_key: Groq API key (read from st.secrets in the UI layer).
    """

    def __init__(self, groq_api_key: str) -> None:
        self._client = Groq(api_key=groq_api_key)
        self.vector_store = VectorStore()
        self.memory = ConversationMemory()

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(
        self,
        question: str,
        filter_sources: Optional[List[str]] = None,
        n_results: int = N_RETRIEVE,
    ) -> RAGAnswer:
        """
        Full RAG pipeline: retrieve → generate → cite.

        Args:
            question:       The user's question.
            filter_sources: Restrict retrieval to these filenames (or None = all).
            n_results:      Number of chunks to retrieve.

        Returns:
            RAGAnswer with answer text, citations list, and source chunks.
        """
        # ── Step 1: Retrieve relevant chunks ──────────────────────────────────
        chunks = self.vector_store.query(
            question, n_results=n_results, filter_sources=filter_sources
        )

        if not chunks:
            no_doc_answer = RAGAnswer(
                answer=(
                    "No indexed documents found. "
                    "Please upload and process at least one PDF first."
                )
            )
            return no_doc_answer

        # ── Step 2: Build context block ───────────────────────────────────────
        context_block = _build_context_block(chunks)

        # ── Step 3: Build message list with memory ────────────────────────────
        messages = self.memory.build_messages(
            system_prompt=SYSTEM_PROMPT,
            context_block=context_block,
            current_question=question,
        )

        # ── Step 4: Call Groq ─────────────────────────────────────────────────
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                temperature=0.2,   # lower temperature → more factual, less creative
            )
            answer_text = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("Groq API error: %s", exc)
            return RAGAnswer(answer=f"LLM error: {exc}", source_chunks=chunks)

        # ── Step 5: Build deterministic citations ─────────────────────────────
        citations = _build_citations(chunks)

        # ── Step 6: Update memory ─────────────────────────────────────────────
        self.memory.add_turn("user", question)
        self.memory.add_turn("assistant", answer_text)
        # Best-effort compression (does nothing if below threshold)
        self.memory.maybe_compress(self._client, MODEL)

        return RAGAnswer(
            answer=answer_text,
            citations=citations,
            source_chunks=chunks,
            retrieval_count=len(chunks),
        )

    def simple_ask(self, prompt: str, system: str = "", max_tokens: int = 1500) -> str:
        """
        Direct Groq call without RAG (used by Study Mode, Agent, Analysis).
        No memory, no retrieval.  Just prompt → answer.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=messages,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def clear_memory(self) -> None:
        self.memory.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context_block(chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Each chunk is labelled with its source and page so that the LLM
    can refer to them (even though we ultimately cite from metadata).
    """
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Chunk {i} | {chunk.source}, Page {chunk.page}]\n{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def _build_citations(chunks: List[RetrievedChunk]) -> List[str]:
    """
    Produce a deduplicated, sorted list of citation strings from chunk metadata.

    Each unique (source, page) pair becomes one citation entry.
    Sorted by source name then page number for consistent display.

    Example output:
        ["annual_report.pdf · p.3", "annual_report.pdf · p.7", "policy.pdf · p.1"]

    This function never calls the LLM — citations are 100% from metadata.
    """
    seen: set = set()
    citations: List[str] = []

    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            citations.append(chunk.citation_label())

    # Sort: primary by filename, secondary by page number
    citations.sort(key=lambda c: (c.split(" · p.")[0], int(c.split(" · p.")[1].rstrip("]"))))
    return citations
