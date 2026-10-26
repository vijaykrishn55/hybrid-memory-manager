from hybrid_memory import HybridMemoryManager, SlidingWindowMemory


def test_window_overflow_retains_exact_last_five(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(window_size=5), user_id="u1", embedder=embedder)
    for i in range(25):
        manager.add_interaction("user", f"unique-message-{i}")

    messages = manager.get_messages()
    assert len(messages) == 5
    assert [item["content"] for item in messages] == [f"unique-message-{i}" for i in range(20, 25)]
    assert len(manager.get_full_history()) == 25


def test_standalone_window_returns_evicted_message():
    from hybrid_memory.models import ConversationMessage

    window = SlidingWindowMemory(2)
    window.append(ConversationMessage(role="user", content="one"))
    window.append(ConversationMessage(role="user", content="two"))
    evicted = window.append(ConversationMessage(role="user", content="three"))
    assert evicted is not None
    assert evicted.content == "one"
    assert [m.content for m in window.get_messages()] == ["two", "three"]


def test_empty_window_is_safe(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(), embedder=embedder)
    assert manager.get_messages() == []
    assert manager.build_context_payload("hello")
