import pytest

from hybrid_memory import HashingEmbedder, MemoryConfig


@pytest.fixture
def config_factory():
    def make(**overrides):
        values = {
            "window_size": 5,
            "context_token_budget": 500,
            "summary_token_budget": 100,
            "semantic_top_k": 3,
            "semantic_similarity_threshold": 0.75,
            "embedding_backend": "hashing",
        }
        values.update(overrides)
        return MemoryConfig(**values)
    return make


@pytest.fixture
def embedder():
    return HashingEmbedder()
