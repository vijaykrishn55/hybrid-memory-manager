from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Role = Literal["system", "user", "assistant", "tool"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
    user_id: str = "default-user"
    session_id: str = "default-session"
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_is_string(cls, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("message content must be a string")
        return value


class SemanticMemoryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: Role
    content: str
    timestamp: datetime = Field(default_factory=utc_now)
    user_id: str = "default-user"
    session_id: str = "default-session"
    token_count: int = Field(default=0, ge=0)
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_size: int = Field(default=10, ge=1)
    context_token_budget: int = Field(default=1500, ge=32)
    summary_token_budget: int = Field(default=300, ge=0)
    semantic_top_k: int = Field(default=3, ge=0)
    semantic_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    embedding_backend: Literal["sentence-transformers", "hashing"] = "sentence-transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
