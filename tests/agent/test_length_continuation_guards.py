"""Length-continuation guards: reasoning-only truncations are not continued; output cap is configurable."""
from types import SimpleNamespace

from agent import conversation_loop as cl


def test_cap_default_and_env(monkeypatch):
    monkeypatch.delenv("HERMES_LENGTH_CONTINUATION_MAX_TOKENS", raising=False)
    assert cl._length_continuation_output_cap() == 32768
    monkeypatch.setenv("HERMES_LENGTH_CONTINUATION_MAX_TOKENS", "8192")
    assert cl._length_continuation_output_cap() == 8192
    monkeypatch.setenv("HERMES_LENGTH_CONTINUATION_MAX_TOKENS", "junk")
    assert cl._length_continuation_output_cap() == 32768


def test_reasoning_only_truncation_continued_exactly_once(monkeypatch):
    """Compromise semantics (master verdict 8): the first reasoning-only truncation gets
    upstream's reasoning-off continuation, a second one does not."""
    monkeypatch.delenv("HERMES_LENGTH_CONTINUATION_REASONING_ONLY", raising=False)
    empty = SimpleNamespace(content="", reasoning_content="lots of thinking")
    # No continuation issued yet → upstream's reasoning-off retry gets its one shot.
    assert cl._length_continuation_worthwhile(empty, []) is True
    assert cl._length_continuation_worthwhile(empty, [], retries=0) is True
    # That shot is spent: a second reasoning-only truncation is thinking exhaustion, not a retry.
    assert cl._length_continuation_worthwhile(empty, [], retries=1) is False
    assert cl._length_continuation_worthwhile(empty, [], retries=2) is False
    # Visible text (this fragment or an earlier one) always continues, at any retry count.
    assert cl._length_continuation_worthwhile(SimpleNamespace(content="partial answer"), []) is True
    assert cl._length_continuation_worthwhile(
        SimpleNamespace(content="partial answer"), [], retries=2
    ) is True
    assert cl._length_continuation_worthwhile(empty, ["earlier visible part"], retries=2) is True
    monkeypatch.setenv("HERMES_LENGTH_CONTINUATION_REASONING_ONLY", "1")
    assert cl._length_continuation_worthwhile(empty, [], retries=2) is True
