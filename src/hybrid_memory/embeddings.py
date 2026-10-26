from __future__ import annotations

import logging
from functools import lru_cache
from typing import Protocol

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

logger = logging.getLogger(__name__)


class TextEmbedder(Protocol):
    name: str

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Small, stateless local vectorizer used for fast startup and offline fallback."""

    name = "hashing"

    def __init__(self, n_features: int = 1024) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            lowercase=True,
        )

    def embed(self, text: str) -> np.ndarray:
        vector = self._vectorizer.transform([text]).toarray()[0].astype(np.float32)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector


@lru_cache(maxsize=2)
def _load_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Local sentence embeddings; downloads model weights on first use if not cached."""

    name = "sentence-transformers"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = _load_sentence_transformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        vector = self._model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        return np.asarray(vector, dtype=np.float32)


def build_embedder(backend: str, model_name: str) -> TextEmbedder:
    if backend == "hashing":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:
        logger.warning(
            "Could not load local sentence-transformer model (%s). Falling back to hashing vectors: %s",
            model_name,
            exc,
        )
        return HashingEmbedder()
