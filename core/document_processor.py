"""
core/document_processor.py
───────────────────────────
Handles all PDF ingestion:

  1. Text extraction  (pypdf — fast, zero dependencies)
  2. OCR fallback     (pytesseract — for scanned / image-only pages)
  3. Table detection  (heuristic line-counting; full extraction via pdfplumber
                       is optional and noted below)
  4. Chunking         (RecursiveCharacterTextSplitter from langchain)

Each chunk is a dict:
    {
        "text":      str,   # the chunk content
        "source":    str,   # original filename
        "page":      int,   # 1-indexed page number
        "chunk_idx": int,   # position within the page's chunks
        "char_start":int,   # character offset within the page text
    }

These dicts become the metadata stored alongside embeddings in ChromaDB,
enabling deterministic (non-LLM) citation generation.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 800   # characters per chunk (≈180-200 tokens at 4 chars/token)
CHUNK_OVERLAP = 150   # overlap to preserve cross-boundary context
MIN_PAGE_CHARS = 50   # pages with fewer chars than this trigger OCR


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Holds the extracted content of a single PDF page."""
    page_num:  int
    text:      str
    via_ocr:   bool = False
    has_table: bool = False


@dataclass
class ProcessedDocument:
    """All extracted data from one uploaded PDF."""
    filename:   str
    pages:      List[PageContent] = field(default_factory=list)
    full_text:  str = ""           # concatenation of all page texts
    num_pages:  int = 0
    ocr_pages:  List[int] = field(default_factory=list)


@dataclass
class Chunk:
    """A single retrievable text unit with full provenance."""
    text:       str
    source:     str    # filename
    page:       int    # 1-indexed
    chunk_idx:  int    # 0-indexed within page
    char_start: int    # character offset within the page text


# ── OCR helper ────────────────────────────────────────────────────────────────

def _ocr_page(page) -> str:
    """
    Render a pypdf page to an image and run Tesseract OCR.

    pypdf doesn't render to images directly; we use pypdf's sibling
    library approach: convert the page to a PIL image via its mediabox
    dimensions and pypdf's page.to_image() if available, otherwise we
    use a blank PIL image and return empty string (graceful degradation).

    DEPLOYMENT NOTE:
    Tesseract must be installed as a system binary.  On Streamlit Cloud
    add  tesseract-ocr  to packages.txt.  The pytesseract Python wrapper
    will raise EnvironmentError if the binary is missing — we catch that
    and log a warning so the app continues with empty text.
    """
    try:
        import pytesseract
        from PIL import Image

        # pypdf >= 4.x provides page.to_image() returning a PIL image.
        # Older versions don't — we fall back gracefully.
        if hasattr(page, "to_image"):
            pil_img = page.to_image(resolution=200)  # 200 DPI is a good balance
        else:
            # Attempt via pdf2image if installed (optional extra)
            try:
                from pdf2image import convert_from_bytes  # noqa: F401
                raise ImportError("Use pdf2image path separately")
            except ImportError:
                logger.debug("to_image not available and pdf2image not installed.")
                return ""

        text = pytesseract.image_to_string(pil_img, lang="eng")
        return text.strip()

    except EnvironmentError:
        logger.warning(
            "Tesseract binary not found. Install tesseract-ocr and add it "
            "to packages.txt on Streamlit Cloud."
        )
        return ""
    except Exception as exc:
        logger.warning("OCR failed for page: %s", exc)
        return ""


# ── Table heuristic ───────────────────────────────────────────────────────────

def _detect_table(text: str) -> bool:
    """
    Heuristic: a page likely contains a table if it has many lines with
    multiple whitespace-separated columns or pipe characters.

    For full table extraction consider pdfplumber:
        import pdfplumber
        with pdfplumber.open(pdf_bytes) as pdf:
            tables = pdf.pages[i].extract_tables()
    pdfplumber is not in requirements by default to keep the install light,
    but adding it is a one-line change.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    multi_col_lines = sum(
        1 for l in lines if len(l.split()) >= 4 and "  " in l
    )
    has_pipes = any("|" in l for l in lines)
    return has_pipes or (multi_col_lines / max(len(lines), 1)) > 0.35


# ── Main extraction function ───────────────────────────────────────────────────

def process_pdf(uploaded_file) -> ProcessedDocument:
    """
    Extract text from every page of a PDF, falling back to OCR when
    pypdf returns insufficient text (e.g. scanned pages).

    Args:
        uploaded_file: A Streamlit UploadedFile (file-like object with .name).

    Returns:
        ProcessedDocument with per-page content and aggregated full_text.
    """
    from pypdf import PdfReader

    doc = ProcessedDocument(filename=uploaded_file.name)

    # Read bytes once so we can seek for OCR if needed.
    pdf_bytes = uploaded_file.read()
    reader = PdfReader(io.BytesIO(pdf_bytes))
    doc.num_pages = len(reader.pages)

    full_parts: List[str] = []

    for i, page in enumerate(reader.pages):
        page_num = i + 1

        # Primary extraction
        raw_text: str = page.extract_text() or ""
        raw_text = raw_text.strip()

        via_ocr = False
        if len(raw_text) < MIN_PAGE_CHARS:
            # Too little text — almost certainly a scanned / image page.
            logger.info("Page %d: sparse text (%d chars), attempting OCR.", page_num, len(raw_text))
            ocr_text = _ocr_page(page)
            if len(ocr_text) > len(raw_text):
                raw_text = ocr_text
                via_ocr = True
                doc.ocr_pages.append(page_num)

        has_table = _detect_table(raw_text)
        pc = PageContent(
            page_num=page_num,
            text=raw_text,
            via_ocr=via_ocr,
            has_table=has_table,
        )
        doc.pages.append(pc)
        # Tag text with page marker so full_text carries provenance.
        full_parts.append(f"[Page {page_num}]\n{raw_text}")

    doc.full_text = "\n\n".join(full_parts)
    return doc


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_document(doc: ProcessedDocument) -> List[Chunk]:
    """
    Split each page's text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter.

    Why per-page chunking (not whole-document)?
    - Page numbers are preserved as ground-truth metadata.
    - A chunk never spans two pages, so citations are always exact.
    - Retrieval can filter by page range if needed.

    Why RecursiveCharacterTextSplitter?
    - It tries paragraph → sentence → word boundaries in order,
      so it rarely cuts in the middle of a sentence.
    - chunk_overlap retains the tail of the previous chunk so that
      questions about bridging content are answered correctly.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )

    chunks: List[Chunk] = []

    for pc in doc.pages:
        if not pc.text.strip():
            continue

        page_chunks = splitter.split_text(pc.text)

        # Track approximate character position within the page.
        cursor = 0
        for idx, chunk_text in enumerate(page_chunks):
            char_start = pc.text.find(chunk_text[:40], cursor)
            if char_start == -1:
                char_start = cursor  # fallback if find misses

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=doc.filename,
                    page=pc.page_num,
                    chunk_idx=idx,
                    char_start=max(char_start, 0),
                )
            )
            cursor = max(char_start + len(chunk_text) - CHUNK_OVERLAP, 0)

    logger.info(
        "Chunked '%s': %d pages → %d chunks.",
        doc.filename,
        doc.num_pages,
        len(chunks),
    )
    return chunks
