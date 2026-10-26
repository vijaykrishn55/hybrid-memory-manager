from hybrid_memory import TokenCounter


def test_token_truncation_stays_within_limit():
    counter = TokenCounter()
    text = "memory systems retain relevant context " * 100
    truncated = counter.truncate(text, 50)
    assert counter.count(truncated) <= 50
    assert truncated


def test_empty_text_count_is_zero():
    assert TokenCounter().count("") == 0
