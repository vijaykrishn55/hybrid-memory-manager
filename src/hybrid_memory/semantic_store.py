from __future__ import annotations

import logging
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .embeddings import TextEmbedder
from .models import ConversationMessage, SemanticMemoryEntry
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "best", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "the", "to",
    "what", "which", "who", "why", "with", "you", "your", "my", "we", "our", "can",
}
_SYNONYMS = {
    "like": "preference", "likes": "preference", "love": "preference", "loves": "preference",
    "prefer": "preference", "prefers": "preference", "preferred": "preference",
    "favorite": "preference", "favourite": "preference", "favorites": "preference",
    "favourites": "preference", "enjoy": "preference", "enjoys": "preference",
}


def _content_terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-z0-9]+", text.lower()))
    normalized = {_SYNONYMS.get(term, term) for term in terms}
    return normalized - _STOP_WORDS


class SemanticMemoryStore:
    """Persistent-ready vector store with user scoping and cosine-based ranking."""

    def __init__(self, embedder: TextEmbedder, similarity_threshold: float = 0.75,
                 token_counter: TokenCounter | None = None) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.counter = token_counter or TokenCounter()
        self._entries: list[SemanticMemoryEntry] = []
        self._lock = threading.RLock()

    def add_message(self, message: ConversationMessage) -> SemanticMemoryEntry:
        if not message.content.strip():
            raise ValueError("cannot store an empty semantic memory")
        vector = np.asarray(self.embedder.embed(message.content), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector = vector / norm
        entry = SemanticMemoryEntry(
            id=str(uuid.uuid4()),
            role=message.role,
            content=message.content,
            timestamp=message.timestamp,
            user_id=message.user_id,
            session_id=message.session_id,
            token_count=message.token_count or self.counter.count(message.content),
            embedding=vector.tolist(),
            metadata=dict(message.metadata),
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def search(self, query: str, user_id: str, top_k: int = 3,
               min_similarity: float | None = None) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            return []
        try:
            query_vector = np.asarray(self.embedder.embed(query), dtype=np.float32).reshape(-1)
            query_norm = float(np.linalg.norm(query_vector))
            if not query_norm:
                return []
            query_vector = query_vector / query_norm
        except Exception as exc:
            logger.warning("Semantic query embedding failed: %s", exc)
            return []

        with self._lock:
            entries = [entry for entry in self._entries if entry.user_id == user_id]

        query_terms = _content_terms(query)
        threshold = self.similarity_threshold if min_similarity is None else min_similarity
        compatible: list[tuple[SemanticMemoryEntry, np.ndarray]] = []
        for entry in entries:
            vector = np.asarray(entry.embedding, dtype=np.float32).reshape(-1)
            if vector.shape == query_vector.shape:
                compatible.append((entry, vector))
        if not compatible:
            return []

        # Dense similarity is calculated as one matrix operation rather than one Python
        # dot-product loop per item, which keeps larger local stores responsive.
        matrix = np.vstack([vector for _, vector in compatible])
        norms = np.linalg.norm(matrix, axis=1)
        matrix = matrix / np.maximum(norms[:, None], 1e-12)
        cosines = np.clip(matrix @ query_vector, -1.0, 1.0)

        scored: list[tuple[float, float, SemanticMemoryEntry]] = []
        for (entry, _), cosine_value in zip(compatible, cosines):
            cosine = float(cosine_value)
            normalized_cosine = (cosine + 1.0) / 2.0
            memory_terms = _content_terms(entry.content)
            overlap = len(query_terms & memory_terms) / max(1, min(len(query_terms), len(memory_terms)))
            # The relevance score combines normalized cosine similarity with a lexical
            # overlap signal, helping short paraphrased queries retrieve explicit facts.
            score = max(normalized_cosine, overlap)
            if score >= threshold:
                scored.append((score, cosine, entry))

        scored.sort(key=lambda row: (row[0], row[2].timestamp), reverse=True)
        results: list[dict[str, Any]] = []
        for score, cosine, entry in scored[:top_k]:
            results.append({
                "id": entry.id,
                "role": entry.role,
                "content": entry.content,
                "timestamp": entry.timestamp.isoformat(),
                "user_id": entry.user_id,
                "session_id": entry.session_id,
                "token_count": entry.token_count,
                "metadata": dict(entry.metadata),
                "similarity_score": score,
                "cosine_similarity": cosine,
            })
        return results

    def entries(self, user_id: str | None = None) -> list[SemanticMemoryEntry]:
        with self._lock:
            selected = self._entries if user_id is None else [e for e in self._entries if e.user_id == user_id]
            return [entry.model_copy(deep=True) for entry in selected]

    def load_entries(self, entries: list[SemanticMemoryEntry]) -> None:
        with self._lock:
            self._entries = [entry.model_copy(deep=True) for entry in entries]

    def forget_user(self, user_id: str) -> int:
        with self._lock:
            before = len(self._entries)
            self._entries = [entry for entry in self._entries if entry.user_id != user_id]
            return before - len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
