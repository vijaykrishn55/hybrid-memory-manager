from __future__ import annotations

import re
from typing import Protocol

from .models import ConversationMessage
from .token_counter import TokenCounter


class Summarizer(Protocol):
    def summarize(
        self, existing_summary: str, new_messages: list[ConversationMessage], max_tokens: int
    ) -> str: ...


_FACT_PATTERNS = (
    r"\bmy name is\b", r"\bi live in\b", r"\bi am from\b", r"\bi'm from\b",
    r"\bfavou?rite\b", r"\bi prefer\b", r"\bi like\b", r"\bi love\b",
    r"\bi am building\b", r"\bi'm building\b", r"\bworking on\b", r"\bproject\b",
    r"\bmy goal\b", r"\bi need\b", r"\bi want\b", r"\bmust\b", r"\bdeadline\b",
    r"\busing\b", r"\buses\b", r"\btimezone\b", r"\bconstraint\b", r"\bbudget\b",
    r"\bprogramming language\b", r"\bremember\b", r"\bdo not\b", r"\bdon't\b",
)


class ExtractiveSummarizer:
    """Deterministic local summarizer that prioritizes explicit facts and preferences."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.counter = token_counter or TokenCounter()

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip(" \t\n•-") for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]

    @staticmethod
    def _score(sentence: str) -> int:
        lowered = sentence.lower()
        return sum(3 for pattern in _FACT_PATTERNS if re.search(pattern, lowered))

    def summarize(
        self, existing_summary: str, new_messages: list[ConversationMessage], max_tokens: int
    ) -> str:
        if max_tokens <= 0:
            return ""

        candidates: list[tuple[int, int, str]] = []
        order = 0
        for sentence in self._sentences(existing_summary):
            candidates.append((self._score(sentence) + 1, order, sentence))
            order += 1
        for message in new_messages:
            for sentence in self._sentences(message.content):
                candidates.append((self._score(sentence), order, sentence))
                order += 1

        # Deduplicate while preferring fact-bearing sentences. Preserve chronological
        # order after selecting the highest-value material that fits the budget.
        best_by_text: dict[str, tuple[int, int, str]] = {}
        for candidate in candidates:
            key = re.sub(r"\W+", " ", candidate[2].lower()).strip()
            if not key:
                continue
            if key not in best_by_text or candidate[0] > best_by_text[key][0]:
                best_by_text[key] = candidate

        ranked = sorted(best_by_text.values(), key=lambda item: (item[0], item[1]), reverse=True)
        selected: list[tuple[int, int, str]] = []
        used_tokens = 0
        for candidate in ranked:
            cost = self.counter.count(candidate[2])
            if used_tokens + cost <= max_tokens:
                selected.append(candidate)
                used_tokens += cost

        if not selected and ranked:
            return self.counter.truncate(ranked[0][2], max_tokens)
        selected.sort(key=lambda item: item[1])
        return " ".join(item[2] for item in selected)


class SummaryBufferedMemory:
    def __init__(self, max_tokens: int = 300, summarizer: Summarizer | None = None,
                 token_counter: TokenCounter | None = None) -> None:
        self.max_tokens = max_tokens
        self.counter = token_counter or TokenCounter()
        self.summarizer = summarizer or ExtractiveSummarizer(self.counter)
        self.summary = ""

    def absorb(self, message: ConversationMessage) -> None:
        try:
            updated = self.summarizer.summarize(self.summary, [message], self.max_tokens)
            self.summary = self.counter.truncate(updated.strip(), self.max_tokens)
        except Exception:
            # Keep the last valid summary if an injected summarizer fails.
            self.summary = self.counter.truncate(self.summary, self.max_tokens)

    def set_summary(self, summary: str) -> None:
        self.summary = self.counter.truncate(summary, self.max_tokens)

    def clear(self) -> None:
        self.summary = ""
