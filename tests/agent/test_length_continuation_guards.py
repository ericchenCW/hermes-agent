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


def test_reasoning_only_truncation_not_continued(monkeypatch):
    monkeypatch.delenv("HERMES_LENGTH_CONTINUATION_REASONING_ONLY", raising=False)
    empty = SimpleNamespace(content="", reasoning_content="lots of thinking")
    assert cl._length_continuation_worthwhile(empty, []) is False
    assert cl._length_continuation_worthwhile(SimpleNamespace(content="partial answer"), []) is True
    assert cl._length_continuation_worthwhile(empty, ["earlier visible part"]) is True
    monkeypatch.setenv("HERMES_LENGTH_CONTINUATION_REASONING_ONLY", "1")
    assert cl._length_continuation_worthwhile(empty, []) is True
