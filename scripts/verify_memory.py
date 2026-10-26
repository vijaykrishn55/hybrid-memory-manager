from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_memory import HashingEmbedder, HybridMemoryManager, MemoryConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the hybrid memory manager end to end.")
    parser.add_argument("--embedding-backend", choices=["hashing", "sentence-transformers"], default="hashing")
    parser.add_argument("--state", type=Path, default=Path("reports/memory_state.json"))
    args = parser.parse_args()

    config = MemoryConfig(
        window_size=4,
        context_token_budget=500,
        summary_token_budget=120,
        semantic_top_k=3,
        semantic_similarity_threshold=0.75,
        embedding_backend=args.embedding_backend,
    )
    manager = HybridMemoryManager(config, user_id="user-alex", session_id="session-a")
    print(f"Embedding backend requested: {args.embedding_backend}")

    manager.add_interaction("user", "Hi, my name is Alex and I live in Seattle. I am building a Python web scraper.")
    manager.add_interaction("assistant", "Nice to meet you, Alex. For a lightweight scraper, consider Requests and BeautifulSoup.")
    manager.add_interaction("user", "I prefer lightweight Python tools and clear examples.")
    manager.add_interaction("assistant", "Understood. I will keep suggestions lightweight and practical.")
    manager.add_interaction("user", "My favorite programming language is Rust.")
    manager.add_interaction("assistant", "I will keep that preference in mind.")

    manager.start_session("session-b", clear_short_term=True)
    query = "What language do I like best?"
    payload = manager.build_context_payload(query)

    output = {
        "active_window_size": len(manager.get_messages()),
        "full_history_messages": len(manager.get_full_history()),
        "running_summary": manager.running_summary,
        "retrieved_memories": [
            {"content": item["content"], "score": item["similarity_score"], "session_id": item["session_id"]}
            for item in manager.semantic_store.search(query, user_id="user-alex", top_k=3)
        ],
        "context_token_count": manager.context_token_count(payload),
        "context_token_budget": config.context_token_budget,
        "payload": payload,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    manager.save(args.state)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    retrieved_rust = any("Rust" in item["content"] for item in output["retrieved_memories"])
    required = [
        ("cross-session Rust recall", retrieved_rust),
        ("context budget respected", output["context_token_count"] <= config.context_token_budget),
        ("full conversation history retained", output["full_history_messages"] == 6),
        ("short-term buffer cleared for new session", output["active_window_size"] == 0),
    ]
    print("\nVerification checks:")
    for name, passed in required:
        print(f"{'PASS' if passed else 'FAIL'}: {name}")
    print(f"State saved to: {args.state}")
    return 0 if all(passed for _, passed in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
