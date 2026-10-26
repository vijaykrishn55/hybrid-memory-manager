from hybrid_memory import HybridMemoryManager


def test_summary_preserves_key_facts_after_window_eviction(config_factory, embedder):
    manager = HybridMemoryManager(
        config_factory(window_size=2, summary_token_budget=100),
        user_id="alex",
        embedder=embedder,
    )
    manager.add_interaction("user", "My name is Alex. I live in Seattle and I am building a Python web scraper.")
    manager.add_interaction("assistant", "You could use Requests and BeautifulSoup.")
    for i in range(8):
        manager.add_interaction("user", f"Unrelated follow-up message {i}.")

    summary = manager.running_summary.lower()
    assert "alex" in summary
    assert "seattle" in summary
    assert "python web scraper" in summary
    assert len(manager.get_messages()) == 2
    assert manager.token_counter.count(manager.running_summary) <= 100


def test_summary_stays_within_configured_token_budget(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(window_size=1, summary_token_budget=20), embedder=embedder)
    for i in range(10):
        manager.add_interaction("user", f"My preference number {i} is to use clear practical examples.")
    assert manager.token_counter.count(manager.running_summary) <= 20
