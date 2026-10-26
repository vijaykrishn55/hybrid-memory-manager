from __future__ import annotations

from collections import deque

from .models import ConversationMessage


class SlidingWindowMemory:
    """Thread-safe manager-owned bounded recent-message buffer; oldest messages leave first."""

    def __init__(self, max_messages: int = 10) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")
        self.max_messages = max_messages
        self._messages: deque[ConversationMessage] = deque()

    def append(self, message: ConversationMessage) -> ConversationMessage | None:
        evicted = None
        self._messages.append(message)
        if len(self._messages) > self.max_messages:
            evicted = self._messages.popleft()
        return evicted

    def get_messages(self) -> list[ConversationMessage]:
        return list(self._messages)

    def load_messages(self, messages: list[ConversationMessage]) -> None:
        self._messages.clear()
        self._messages.extend(messages[-self.max_messages:])

    def clear(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)
