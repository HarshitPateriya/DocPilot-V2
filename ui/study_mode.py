"""
ui/study_mode.py
─────────────────
Study Mode tab: Quiz, Flashcards, Interview Questions.

All three features use RAG retrieval (not full-text) so they scale to
multi-hundred-page documents.  The retrieval query is tuned per feature:
  - Quiz: "key facts definitions concepts"
  - Flashcards: "terms definitions vocabulary"
  - Interview: "processes methods decisions outcomes"

JSON parsing is robust: strips markdown fences, retries once on failure.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

import streamlit as st

from core.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


def render_study_mode() -> None:
    docs: dict = st.session_state.docs
    rag: RAGEngine = st.session_state.rag

    st.subheader("🧠 Smart Study Mode")

    if not docs:
        st.info("Upload at least one PDF to use Study Mode.")
        return

    doc_choice = st.selectbox("Choose document", list(docs.keys()))
    tab1, tab2, tab3 = st.tabs(["📝 Quiz", "🃏 Flashcards", "🎯 Interview Questions"])

    with tab1:
        _render_quiz(doc_choice, rag)
    with tab2:
        _render_flashcards(doc_choice, rag)
    with tab3:
        _render_interview(doc_choice, rag)


# ── Quiz ──────────────────────────────────────────────────────────────────────

def _render_quiz(filename: str, rag: RAGEngine) -> None:
    num_q = st.slider("Number of questions", 3, 10, 5, key="quiz_n")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], key="quiz_diff")

    if st.button("Generate Quiz", key="gen_quiz", type="primary"):
        with st.spinner("Generating quiz…"):
            context = _retrieve_context(
                filename, rag,
                query="key facts definitions concepts important points",
                n=12,
            )
            prompt = (
                f"Create exactly {num_q} {difficulty.lower()}-difficulty multiple-choice "
                f"questions based ONLY on the following document content.\n\n"
                f"Return ONLY a valid JSON array (no markdown, no preamble):\n"
                f'[{{"q":"question text","options":["A. option","B. option","C. option","D. option"],'
                f'"answer":"A","explanation":"brief reason"}}]\n\n'
                f"Document: {filename}\n\n{context}"
            )
            try:
                raw = rag.simple_ask(prompt)
                questions = _safe_parse_json(raw)
                _render_quiz_ui(questions)
            except Exception as exc:
                st.error(f"Quiz generation failed: {exc}")
                st.text(raw[:500] if "raw" in dir() else "No response")


def _render_quiz_ui(questions: List[Dict]) -> None:
    if not questions:
        st.warning("No questions generated. Try a different document or difficulty.")
        return
    st.success(f"Generated {len(questions)} questions.")
    for i, item in enumerate(questions, 1):
        with st.expander(f"Q{i}: {item.get('q', 'Question')}"):
            for opt in item.get("options", []):
                st.write(f"• {opt}")
            st.success(f"✅ Correct answer: {item.get('answer', 'N/A')}")
            if item.get("explanation"):
                st.info(f"💡 {item['explanation']}")
    all_text = "\n\n".join(
        f"Q{i}: {q.get('q','')}\n"
        + "\n".join(q.get("options", []))
        + f"\nAnswer: {q.get('answer','')}\nExplanation: {q.get('explanation','')}"
        for i, q in enumerate(questions, 1)
    )
    st.download_button("⬇ Download Quiz", all_text, "quiz.txt")


# ── Flashcards ────────────────────────────────────────────────────────────────

def _render_flashcards(filename: str, rag: RAGEngine) -> None:
    num_f = st.slider("Number of flashcards", 5, 20, 10, key="flash_n")

    if st.button("Generate Flashcards", key="gen_flash", type="primary"):
        with st.spinner("Generating flashcards…"):
            context = _retrieve_context(
                filename, rag,
                query="terms definitions vocabulary key concepts",
                n=12,
            )
            prompt = (
                f"Create exactly {num_f} flashcards from the following document content.\n\n"
                f"Return ONLY a valid JSON array (no markdown, no preamble):\n"
                f'[{{"front":"term or question","back":"definition or answer"}}]\n\n'
                f"Document: {filename}\n\n{context}"
            )
            try:
                raw = rag.simple_ask(prompt)
                cards = _safe_parse_json(raw)
                _render_flashcards_ui(cards)
            except Exception as exc:
                st.error(f"Flashcard generation failed: {exc}")


def _render_flashcards_ui(cards: List[Dict]) -> None:
    if not cards:
        st.warning("No flashcards generated.")
        return
    st.success(f"Generated {len(cards)} flashcards.")
    for i, card in enumerate(cards, 1):
        with st.expander(f"Card {i}: {card.get('front', 'Term')}"):
            st.markdown(f"**{card.get('back', 'Definition')}**")
    all_text = "\n\n".join(
        f"Q: {c.get('front','')}\nA: {c.get('back','')}" for c in cards
    )
    st.download_button("⬇ Download Flashcards", all_text, "flashcards.txt")


# ── Interview Questions ───────────────────────────────────────────────────────

def _render_interview(filename: str, rag: RAGEngine) -> None:
    level = st.selectbox(
        "Seniority level",
        ["Junior", "Mid-level", "Senior", "Expert"],
        key="interview_level",
    )
    num_i = st.slider("Number of questions", 3, 10, 5, key="interview_n")

    if st.button("Generate Interview Questions", key="gen_interview", type="primary"):
        with st.spinner("Crafting questions…"):
            context = _retrieve_context(
                filename, rag,
                query="processes methods architecture decisions trade-offs",
                n=12,
            )
            prompt = (
                f"Generate {num_i} {level}-level technical interview questions "
                f"based ONLY on the following document content.\n\n"
                f"Return ONLY a valid JSON array (no markdown, no preamble):\n"
                f'[{{"question":"...","ideal_answer":"..."}}]\n\n'
                f"Document: {filename}\n\n{context}"
            )
            try:
                raw = rag.simple_ask(prompt)
                qs = _safe_parse_json(raw)
                _render_interview_ui(qs)
            except Exception as exc:
                st.error(f"Interview question generation failed: {exc}")


def _render_interview_ui(qs: List[Dict]) -> None:
    if not qs:
        st.warning("No questions generated.")
        return
    st.success(f"Generated {len(qs)} interview questions.")
    for i, item in enumerate(qs, 1):
        with st.expander(f"Q{i}: {item.get('question', 'Question')}"):
            st.write("**Ideal answer:**")
            st.write(item.get("ideal_answer", ""))
    all_text = "\n\n".join(
        f"Q: {q.get('question','')}\nA: {q.get('ideal_answer','')}" for q in qs
    )
    st.download_button("⬇ Download Questions", all_text, "interview_questions.txt")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _retrieve_context(filename: str, rag: RAGEngine, query: str, n: int) -> str:
    """Retrieve top-n chunks for a document and format as a context block."""
    chunks = rag.vector_store.query(query, n_results=n, filter_sources=[filename])
    if chunks:
        return "\n\n".join(f"[Page {c.page}] {c.text}" for c in chunks)
    # Fallback: first 30k chars of full text
    doc = st.session_state.docs.get(filename)
    return doc.full_text[:30_000] if doc else ""


def _safe_parse_json(raw: str) -> List[Any]:
    """
    Parse a JSON array from LLM output.
    Strips markdown code fences before parsing.
    Raises ValueError with context if parsing fails.
    """
    # Remove ```json ... ``` or ``` ... ``` fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Some models add trailing commas; attempt a fix
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Try extracting just the array portion
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Could not parse JSON: {exc}\nRaw (first 300 chars): {raw[:300]}")
