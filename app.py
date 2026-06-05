"""
app.py
───────
DocPilot v2 — Production RAG Document Assistant.

Entry point.  Thin orchestrator: reads the mode from the sidebar,
routes to the correct UI module.  All business logic lives in core/.

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → connect to Streamlit Cloud.
    Set GROQ_API_KEY in Streamlit Cloud secrets.
    Ensure packages.txt (tesseract-ocr) is committed.
"""

import logging

import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocPilot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (matches original DocPilot aesthetic) ──────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

  html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
  h1, h2, h3                  { font-family: 'Space Mono', monospace !important; }

  /* Soften the sidebar background */
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Style retrieved source chunks */
  .source-chunk {
      background: #1e293b;
      border-left: 3px solid #3b82f6;
      padding: 0.6rem 1rem;
      border-radius: 0 6px 6px 0;
      font-size: 0.85rem;
      margin-bottom: 0.5rem;
  }

  /* Citation pill */
  .citation {
      background: #1e3a5f;
      color: #7dd3fc;
      font-size: 0.72rem;
      padding: 2px 7px;
      border-radius: 10px;
      font-weight: 600;
      margin: 0 2px;
  }

  /* Agent result card */
  .agent-card {
      background: #0f172a;
      border: 1px solid #1e293b;
      border-radius: 10px;
      padding: 1.2rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── UI imports ────────────────────────────────────────────────────────────────
from ui.sidebar      import render_sidebar
from ui.chat_mode    import render_chat_mode
from ui.summary_mode import render_summary_mode
from ui.agent_mode   import render_agent_mode
from ui.study_mode   import render_study_mode
from ui.multidoc_mode import render_multidoc_mode

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 DocPilot")
st.caption(
    "Production RAG · Deterministic Citations · "
    "Conversational Memory · OCR · Multi-Document Intelligence"
)

# ── Sidebar (uploads + mode selection) ───────────────────────────────────────
mode = render_sidebar()

# ── Route to selected mode ────────────────────────────────────────────────────
if not st.session_state.get("docs"):
    st.info(
        "⬅ Upload one or more PDFs from the sidebar to get started.\n\n"
        "DocPilot will extract text (with OCR fallback), chunk your documents, "
        "embed them into a local ChromaDB vector store, and enable RAG-powered Q&A "
        "with deterministic page-level citations."
    )
else:
    if mode == "💬 Chat":
        render_chat_mode()
    elif mode == "📋 Summary":
        render_summary_mode()
    elif mode == "🤖 Agent":
        render_agent_mode()
    elif mode == "🧠 Study Mode":
        render_study_mode()
    elif mode == "🔍 Multi-Doc Analysis":
        render_multidoc_mode()
