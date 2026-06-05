"""
core/embeddings.py
──────────────────
Provides the EmbeddingFunction used by ChromaDB.

Strategy (automatic fallback chain):
  1. sentence-transformers  →  all-MiniLM-L6-v2  (semantic, best quality)
  2. TF-IDF via scikit-learn                      (keyword, offline fallback)

The fallback is triggered when HuggingFace is unreachable (e.g. sandboxed
environments, Streamlit Cloud cold starts with no cache).

Why not Chroma's built-in ONNX embedding?
  Chroma's default embedding downloads a model file whose SHA256 check has been
  observed to fail in certain network environments.  Using sentence-transformers
  directly is more reliable and gives us explicit control over caching.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
from chromadb.api.types import Documents, Embeddings, EmbeddingFunction
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# ── TF-IDF fallback ────────────────────────────────────────────────────────────

class TFIDFEmbeddingFunction(EmbeddingFunction):
    """
    Keyword-based embedding using TF-IDF.

    Not semantically rich (cannot match synonyms), but:
    - Works fully offline.
    - Zero model download.
    - Deterministic and fast.

    The vectorizer is fitted lazily on the first batch of documents it sees,
    then frozen.  Query embeddings reuse the same fitted vocabulary.

    dim: Number of TF-IDF features (== embedding dimension).
         Larger values retain more vocabulary; ChromaDB handles any dimension.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self._vectorizer = TfidfVectorizer(
            max_features=dim,
            sublinear_tf=True,          # log(1+tf) dampens high-frequency terms
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),         # unigrams + bigrams improve recall
        )
        self._fitted = False

    # Called automatically by ChromaDB when adding or querying.
    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            self._vectorizer.fit(input)
            self._fitted = True

        matrix = self._vectorizer.transform(input).toarray().astype(np.float32)

        # L2-normalise so cosine distance == euclidean distance in unit-sphere.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        return matrix.tolist()


# ── Sentence-Transformers embedding ───────────────────────────────────────────

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Semantic embedding using sentence-transformers/all-MiniLM-L6-v2.

    Model details:
    - 384-dimensional embeddings.
    - ~80 MB download, cached in ~/.cache/torch/sentence_transformers/.
    - Fast inference on CPU (~50ms per batch of 32 chunks).
    - Trained on 1B sentence pairs; strong general-purpose retrieval.

    DEPLOYMENT NOTE:
    On Streamlit Cloud the model downloads on first run.  Add a st.spinner()
    in the UI layer to inform the user.  Subsequent runs use the local cache.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        vectors = self._model.encode(
            list(input),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalised; matches cosine similarity
        )
        return vectors.tolist()


# ── Public factory ─────────────────────────────────────────────────────────────

def get_embedding_function() -> EmbeddingFunction:
    """
    Return the best available embedding function.

    Tries SentenceTransformer first; falls back to TF-IDF if import or model
    download fails.  The result is cached in the module-level variable so the
    model is only loaded once per Python process.
    """
    global _cached_ef
    if _cached_ef is not None:
        return _cached_ef

    try:
        ef = SentenceTransformerEmbeddingFunction()
        # Warm-up call to trigger any download errors early.
        ef(["warmup"])
        logger.info("Using SentenceTransformer embeddings (all-MiniLM-L6-v2).")
        _cached_ef = ef
    except Exception as exc:
        logger.warning(
            "SentenceTransformer unavailable (%s). Falling back to TF-IDF.", exc
        )
        _cached_ef = TFIDFEmbeddingFunction()

    return _cached_ef


_cached_ef: EmbeddingFunction | None = None
