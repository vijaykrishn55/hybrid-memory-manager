from hybrid_memory import HybridMemoryManager, SemanticMemoryStore
from hybrid_memory.models import ConversationMessage


def test_semantic_memory_retrieves_fact_across_sessions(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(window_size=3), user_id="user-1", session_id="session-a", embedder=embedder)
    manager.add_interaction("user", "My favorite programming language is Rust.")
    manager.add_interaction("assistant", "I will remember that you prefer Rust.")

    manager.start_session("session-b", clear_short_term=True)
    hits = manager.semantic_store.search("What language do I like best?", user_id="user-1", top_k=3)
    assert any("Rust" in hit["content"] for hit in hits)
    assert all(hit["user_id"] == "user-1" for hit in hits)
    assert all(hit["session_id"] == "session-a" for hit in hits)


def test_memories_are_scoped_by_user(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(), user_id="user-a", embedder=embedder)
    manager.add_interaction("user", "My favorite programming language is Rust.")
    hits = manager.semantic_store.search("favorite programming language Rust", user_id="user-b")
    assert hits == []


def test_empty_memory_and_empty_query_are_safe(embedder):
    store = SemanticMemoryStore(embedder, similarity_threshold=0.75)
    assert store.search("hello", user_id="u1") == []
    assert store.search("", user_id="u1") == []
    assert store.search("hello", user_id="u1", top_k=0) == []


def test_semantic_write_rejects_empty_content(embedder):
    store = SemanticMemoryStore(embedder)
    msg = ConversationMessage(role="user", content=" ", user_id="u1")
    try:
        store.add_message(msg)
        assert False, "expected empty content to be rejected"
    except ValueError as exc:
        assert "empty" in str(exc)
