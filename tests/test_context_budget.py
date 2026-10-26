from hybrid_memory import HybridMemoryManager


def test_context_payload_respects_500_token_budget(config_factory, embedder):
    manager = HybridMemoryManager(
        config_factory(window_size=10, context_token_budget=500, summary_token_budget=100),
        user_id="u1",
        embedder=embedder,
    )
    for i in range(15):
        manager.add_interaction("user" if i % 2 == 0 else "assistant", f"Message {i}: " + ("important context " * 45))

    payload = manager.build_context_payload("Summarize the key decision from the discussion.")
    assert manager.context_token_count(payload) <= 500
    assert payload[0]["role"] == "system"
    assert payload[-1] == {"role": "user", "content": "Summarize the key decision from the discussion."}


def test_already_recorded_current_query_is_not_duplicated(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(), embedder=embedder)
    manager.add_interaction("user", "What is my project?")
    payload = manager.build_context_payload("What is my project?")
    assert sum(1 for item in payload if item["role"] == "user" and item["content"] == "What is my project?") == 1


def test_large_query_is_trimmed_when_it_exceeds_budget(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(context_token_budget=80, window_size=2), embedder=embedder)
    query = "question " * 300
    payload = manager.build_context_payload(query)
    assert manager.context_token_count(payload) <= 80
