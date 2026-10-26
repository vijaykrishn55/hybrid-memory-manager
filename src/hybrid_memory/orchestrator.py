from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from .context_assembler import ContextAssembler
from .embeddings import TextEmbedder, build_embedder
from .models import ConversationMessage, MemoryConfig, SemanticMemoryEntry
from .semantic_store import SemanticMemoryStore
from .sliding_window import SlidingWindowMemory
from .summary_buffer import SummaryBufferedMemory, Summarizer
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)


class HybridMemoryManager:
    """Orchestrates short-term, summary-buffered, and semantic long-term memory."""

    STATE_VERSION = 1

    def __init__(self, config: MemoryConfig | None = None, *, user_id: str = "default-user",
                 session_id: str = "session-1", embedder: TextEmbedder | None = None,
                 summarizer: Summarizer | None = None, token_counter: TokenCounter | None = None) -> None:
        self.config = config or MemoryConfig()
        self.user_id = user_id
        self.session_id = session_id
        self.token_counter = token_counter or TokenCounter()
        self.embedder = embedder or build_embedder(self.config.embedding_backend, self.config.embedding_model)
        self.window = SlidingWindowMemory(self.config.window_size)
        self.summary_buffer = SummaryBufferedMemory(
            max_tokens=self.config.summary_token_budget,
            summarizer=summarizer,
            token_counter=self.token_counter,
        )
        self.semantic_store = SemanticMemoryStore(
            self.embedder,
            similarity_threshold=self.config.semantic_similarity_threshold,
            token_counter=self.token_counter,
        )
        self.conversation_history: list[ConversationMessage] = []
        self.last_errors: list[str] = []
        self._lock = threading.RLock()
        self._assembler = ContextAssembler(self.config, self.token_counter)

    @property
    def running_summary(self) -> str:
        return self.summary_buffer.summary

    @property
    def raw_history(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.window.get_messages()]

    def add_interaction(self, role: str, content: str, *, metadata: dict[str, Any] | None = None) -> ConversationMessage:
        """Append one message to full history, the active window, and semantic memory."""
        message = ConversationMessage(
            role=role,
            content=content,
            user_id=self.user_id,
            session_id=self.session_id,
            token_count=self.token_counter.count(content),
            metadata=metadata or {},
        )
        with self._lock:
            self.conversation_history.append(message.model_copy(deep=True))
            evicted = self.window.append(message)
            if evicted is not None:
                self.summary_buffer.absorb(evicted)
            try:
                self.semantic_store.add_message(message)
            except Exception as exc:
                self.last_errors.append(f"semantic_write: {type(exc).__name__}: {exc}")
                logger.warning("Could not write semantic memory; conversation log was retained: %s", exc)
        return message.model_copy(deep=True)

    async def aadd_interaction(self, role: str, content: str, *,
                               metadata: dict[str, Any] | None = None) -> ConversationMessage:
        """Non-blocking entry point for async applications and concurrent writers."""
        return await asyncio.to_thread(self.add_interaction, role, content, metadata=metadata)

    def get_messages(self) -> list[dict[str, Any]]:
        with self._lock:
            return [message.model_dump(mode="json") for message in self.window.get_messages()]

    def get_full_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [message.model_dump(mode="json") for message in self.conversation_history]

    def clear_short_term(self) -> None:
        """Clear only the active window. Summary, full log, and long-term memories remain."""
        with self._lock:
            self.window.clear()

    def start_session(self, session_id: str, *, clear_short_term: bool = True) -> None:
        with self._lock:
            self.session_id = session_id
            if clear_short_term:
                self.window.clear()

    def build_context_payload(self, current_query: str | None = None) -> list[dict[str, str]]:
        with self._lock:
            summary = self.running_summary
            recent = self.window.get_messages()
        hits: list[dict[str, Any]] = []
        if current_query and self.config.semantic_top_k:
            try:
                hits = self.semantic_store.search(
                    current_query,
                    user_id=self.user_id,
                    top_k=self.config.semantic_top_k,
                    min_similarity=self.config.semantic_similarity_threshold,
                )
            except Exception as exc:
                self.last_errors.append(f"semantic_search: {type(exc).__name__}: {exc}")
                logger.warning("Semantic retrieval failed; building context without retrieved memories: %s", exc)
        return self._assembler.build(summary, hits, recent, current_query, self.user_id, self.session_id)

    async def abuild_context_payload(self, current_query: str | None = None) -> list[dict[str, str]]:
        return await asyncio.to_thread(self.build_context_payload, current_query)

    def context_token_count(self, payload: list[dict[str, str]]) -> int:
        return self.token_counter.count_messages(payload)

    def save(self, path: str | Path) -> Path:
        """Persist active window, complete history, summary, and semantic vectors atomically."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": self.STATE_VERSION,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "config": self.config.model_dump(mode="json"),
                "embedding_backend_used": getattr(self.embedder, "name", self.config.embedding_backend),
                "window": [message.model_dump(mode="json") for message in self.window.get_messages()],
                "conversation_history": [message.model_dump(mode="json") for message in self.conversation_history],
                "running_summary": self.running_summary,
                "semantic_entries": [entry.model_dump(mode="json") for entry in self.semantic_store.entries()],
            }
            temp_path = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_path.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path, *, embedder: TextEmbedder | None = None,
             summarizer: Summarizer | None = None, token_counter: TokenCounter | None = None) -> "HybridMemoryManager":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("version") != cls.STATE_VERSION:
            raise ValueError(f"Unsupported memory state version: {payload.get('version')}")
        config_values = dict(payload["config"])
        if embedder is None:
            actual_backend = payload.get("embedding_backend_used")
            if actual_backend in {"hashing", "sentence-transformers"}:
                config_values["embedding_backend"] = actual_backend
        config = MemoryConfig.model_validate(config_values)
        manager = cls(
            config,
            user_id=payload.get("user_id", "default-user"),
            session_id=payload.get("session_id", "session-1"),
            embedder=embedder,
            summarizer=summarizer,
            token_counter=token_counter,
        )
        with manager._lock:
            manager.window.load_messages([
                ConversationMessage.model_validate(item) for item in payload.get("window", [])
            ])
            manager.conversation_history = [
                ConversationMessage.model_validate(item)
                for item in payload.get("conversation_history", [])
            ]
            manager.summary_buffer.set_summary(payload.get("running_summary", ""))
            manager.semantic_store.load_entries([
                SemanticMemoryEntry.model_validate(item)
                for item in payload.get("semantic_entries", [])
            ])
        return manager

    def forget_user_data(self) -> None:
        """Remove all in-memory conversation and semantic records for this user."""
        with self._lock:
            self.window.clear()
            self.conversation_history.clear()
            self.summary_buffer.clear()
            self.semantic_store.forget_user(self.user_id)
            self.last_errors.clear()


# Keep a name aligned with the assessment's orchestration architecture.
MemoryOrchestrator = HybridMemoryManager
