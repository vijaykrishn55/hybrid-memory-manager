import asyncio

from hybrid_memory import HybridMemoryManager


def test_50_concurrent_async_writes_have_no_corruption(config_factory, embedder):
    manager = HybridMemoryManager(config_factory(window_size=10), user_id="concurrent-u", embedder=embedder)

    async def run_writes():
        await asyncio.gather(*[
            manager.aadd_interaction("user", f"concurrent-message-{i}", metadata={"sequence": i})
            for i in range(50)
        ])

    asyncio.run(run_writes())
    history = manager.get_full_history()
    assert len(history) == 50
    assert len({item["content"] for item in history}) == 50
    assert all(item["metadata"].get("sequence") is not None for item in history)
    assert len(manager.get_messages()) == 10
    assert not manager.last_errors
