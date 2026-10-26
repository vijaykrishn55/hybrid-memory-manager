from __future__ import annotations

import re
from typing import Any


_FALLBACK_RE = re.compile(r"\s+|[^\W_]+|[^\w\s]", re.UNICODE)


class TokenCounter:
    """Model-aware token counter with a deterministic fallback when tiktoken is absent."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.mode = "regex-fallback"
        self._encoding: Any = None
        try:
            import tiktoken  # type: ignore

            self._encoding = tiktoken.get_encoding(encoding_name)
            self.mode = f"tiktoken:{encoding_name}"
        except Exception:
            self._encoding = None

    def encode(self, text: str) -> list[Any]:
        if self._encoding is not None:
            return list(self._encoding.encode(text))
        return _FALLBACK_RE.findall(text)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0 or not text:
            return ""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text
        if self._encoding is not None:
            return self._encoding.decode(tokens[:max_tokens])
        return "".join(tokens[:max_tokens]).rstrip()

    def count_messages(self, messages: list[dict[str, str]]) -> int:
        # A small per-message allowance approximates role and boundary overhead.
        return 2 + sum(4 + self.count(m.get("role", "")) + self.count(m.get("content", ""))
                       for m in messages)
