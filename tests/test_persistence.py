from hybrid_memory import HybridMemoryManager


def test_save_and_reload_preserve_window_summary_and_semantic_store(tmp_path, config_factory, embedder):
    path = tmp_path / "memory_state.json"
    manager = HybridMemoryManager(config_factory(window_size=2), user_id="persist-u", session_id="session-a", embedder=embedder)
    manager.add_interaction("user", "My name is Alex and I live in Seattle.")
    manager.add_interaction("assistant", "Nice to meet you, Alex.")
    manager.add_interaction("user", "My favorite programming language is Rust.")
    manager.save(path)

    reloaded = HybridMemoryManager.load(path, embedder=embedder)
    assert reloaded.user_id == "persist-u"
    assert reloaded.session_id == "session-a"
    assert reloaded.get_messages() == manager.get_messages()
    assert len(reloaded.get_full_history()) == 3
    assert len(reloaded.semantic_store) == 3
    hits = reloaded.semantic_store.search("What language do I like best?", user_id="persist-u")
    assert any("Rust" in item["content"] for item in hits)


def test_start_new_session_keeps_long_term_memory_and_full_log(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(), user_id="u1", session_id="a", embedder=embedder)
    manager.add_interaction("user", "I live in Seattle and my favorite programming language is Rust.")
    manager.start_session("b")
    assert manager.get_messages() == []
    assert len(manager.get_full_history()) == 1
    assert len(manager.semantic_store) == 1


def test_forget_user_data_clears_all_in_memory_records(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(), user_id="u1", embedder=embedder)
    manager.add_interaction("user", "Remember my secret project name Alpha.")
    manager.forget_user_data()
    assert manager.get_messages() == []
    assert manager.get_full_history() == []
    assert manager.running_summary == ""
    assert len(manager.semantic_store) == 0
