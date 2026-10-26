# Hybrid Memory Manager

A local-first memory system for LLM applications. It combines a bounded sliding window, a compact running summary, semantic long-term memory, and token-budgeted context assembly. It does not require an OpenAI API key.

## Features

- FIFO sliding-window memory with configurable message limits.
- Full conversation log with timestamps, roles, token counts, and metadata.
- Extractive summary buffering that prioritizes facts, preferences, goals, and constraints.
- Local vector embeddings, cosine-based retrieval, and lexical overlap support.
- Cross-session memory scoped by user ID.
- Context assembly with a strict token budget and safe handling of empty or failed retrieval.
- JSON persistence, state reload, selective data forgetting, and async writes.
- Tests for overflow, summaries, retrieval, persistence, token limits, and concurrent writes.

## Install

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Verify

```bash
python -m pytest
python scripts/verify_memory.py
```

To use the local SentenceTransformers model for stronger paraphrase retrieval:

```bash
python scripts/verify_memory.py --embedding-backend sentence-transformers
```

The model weights are downloaded on first use. The default verification mode uses a lightweight local hashing vectorizer and does not need model downloads.

## Inspect a prompt payload

After running verification, inspect a new assembled context:

```bash
python scripts/demo_query.py "What language do I like best?"
```

## Basic usage

```python
from hybrid_memory import HybridMemoryManager, MemoryConfig

memory = HybridMemoryManager(
    MemoryConfig(window_size=10, context_token_budget=500),
    user_id="user-123",
    session_id="session-a",
)
memory.add_interaction("user", "My favorite programming language is Rust.")
payload = memory.build_context_payload("What language do I like best?")
memory.save("memory_state.json")
```

Use the same `user_id` across sessions for long-term recall. `start_session(..., clear_short_term=True)` clears the active window while retaining the summary, complete history, and semantic store. Protect persisted JSON files as they contain conversation data. The default summary is deterministic and extractive, so it works without a hosted LLM; a custom summarizer can be injected through the `Summarizer` interface.
