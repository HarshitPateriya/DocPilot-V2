"""
core/vector_store.py
─────────────────────
Manages all interactions with ChromaDB.

Design decisions:
  - EphemeralClient (in-memory): no filesystem writes, no persistence across
    Streamlit restarts.  This is intentional: PDFs are re-uploaded each session.
    To add persistence, swap to chromadb.PersistentClient(path=".chroma_db").

  - One collection per session (named "docpilot_session"): all documents for
    the current session share one collection.  Source filename + page number
    in metadata is what separates them.

  - IDs: "{filename}::{page}::{chunk_idx}" — globally unique within a session.

  - n_results for retrieval: 5 chunks by default.  Each chunk is ~800 chars,
    so 5 chunks ≈ 4000 chars of context, well within Llama 3.3's 128k window.

DEPLOYMENT NOTE:
  ChromaDB pulls in grpcio, onnxruntime, and kubernetes as transitive deps.
  These are all pip-installable but some (onnxruntime) are large (~200 MB).
  On Streamlit Cloud this is fine; on Docker you may want a slim base image.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import chromadb
from chromadb import EphemeralClient

from core.document_processor import Chunk
from core.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

COLLECTION_NAME = "docpilot_session"
DEFAULT_N_RESULTS = 5


# ── Result model ─────────────────────────────────────────────────────────────

class RetrievedChunk:
    """
    A single chunk returned by a similarity search, enriched with metadata.
    This is the object that drives deterministic citations — no LLM involved.
    """

    __slots__ = ("text", "source", "page", "chunk_idx", "distance")

    def __init__(
        self,
        text: str,
        source: str,
        page: int,
        chunk_idx: int,
        distance: float,
    ) -> None:
        self.text      = text
        self.source    = source
        self.page      = page
        self.chunk_idx = chunk_idx
        self.distance  = distance   # lower = more similar (cosine distance)

    @property
    def relevance_score(self) -> float:
        """Convert cosine distance [0, 2] to a 0-1 relevance score."""
        return round(max(0.0, 1.0 - self.distance / 2.0), 3)

    def citation_label(self) -> str:
        """Human-readable citation, e.g. '[research_paper.pdf · p.4]'."""
        return f"[{self.source} · p.{self.page}]"

    def __repr__(self) -> str:
        return (
            f"RetrievedChunk(source={self.source!r}, page={self.page}, "
            f"score={self.relevance_score})"
        )


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """
    Thin wrapper around a ChromaDB EphemeralClient + one Collection.

    Lifecycle:
        store = VectorStore()
        store.add_chunks(chunks)           # from document_processor.chunk_document()
        results = store.query("question")  # returns List[RetrievedChunk]
        store.delete_document("file.pdf")  # remove one doc's chunks
        store.reset()                      # wipe everything
    """

    def __init__(self) -> None:
        self._client: chromadb.ClientAPI = EphemeralClient()
        self._ef = get_embedding_function()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},  # use cosine similarity
        )
        self._indexed_sources: Dict[str, int] = {}  # filename → chunk count
        logger.info("VectorStore initialised (EphemeralClient).")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Upsert a list of Chunk objects into the collection.

        Uses upsert (not add) so re-processing the same file is idempotent.
        Batches of 100 avoid memory spikes with large PDFs.
        """
        if not chunks:
            return

        BATCH = 100
        for start in range(0, len(chunks), BATCH):
            batch = chunks[start : start + BATCH]

            ids       = [_make_id(c) for c in batch]
            documents = [c.text for c in batch]
            metadatas = [
                {
                    "source":    c.source,
                    "page":      c.page,
                    "chunk_idx": c.chunk_idx,
                    "char_start":c.char_start,
                }
                for c in batch
            ]

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        source_name = chunks[0].source
        self._indexed_sources[source_name] = len(chunks)
        logger.info("Upserted %d chunks for '%s'.", len(chunks), source_name)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        n_results: int = DEFAULT_N_RESULTS,
        filter_sources: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Find the n_results most relevant chunks for a question.

        Args:
            question:       Natural language query.
            n_results:      How many chunks to return.
            filter_sources: If provided, restrict search to these filenames.

        Returns:
            List of RetrievedChunk sorted by relevance (most relevant first).
        """
        total = self._collection.count()
        if total == 0:
            return []

        n_results = min(n_results, total)

        where: Optional[Dict] = None
        if filter_sources:
            if len(filter_sources) == 1:
                where = {"source": {"$eq": filter_sources[0]}}
            else:
                where = {"source": {"$in": filter_sources}}

        kwargs: Dict = dict(
            query_texts=[question],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        chunks: List[RetrievedChunk] = []
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for text, meta, dist in zip(docs, metas, distances):
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=meta["source"],
                    page=int(meta["page"]),
                    chunk_idx=int(meta["chunk_idx"]),
                    distance=dist,
                )
            )

        return chunks

    # ── Management ────────────────────────────────────────────────────────────

    def delete_document(self, filename: str) -> None:
        """Remove all chunks belonging to a specific document."""
        self._collection.delete(where={"source": {"$eq": filename}})
        self._indexed_sources.pop(filename, None)
        logger.info("Deleted all chunks for '%s'.", filename)

    def reset(self) -> None:
        """Drop and recreate the collection (wipes all documents)."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._indexed_sources.clear()
        logger.info("VectorStore reset.")

    @property
    def indexed_sources(self) -> Dict[str, int]:
        """Dict mapping each indexed filename to its chunk count."""
        return dict(self._indexed_sources)

    @property
    def total_chunks(self) -> int:
        return self._collection.count()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_id(chunk: Chunk) -> str:
    """
    Generate a deterministic, unique ID for a chunk.
    Format: "filename::page::chunk_idx"
    Colons inside filenames are replaced with underscores.
    """
    safe_source = chunk.source.replace("::", "__")
    return f"{safe_source}::{chunk.page}::{chunk.chunk_idx}"
