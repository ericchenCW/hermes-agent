"""Truncation-path fixes: side-channel reasoning exhaustion, honest attempt
counts, Chinese operator-facing messages, and the streaming repetition guard.

Regression cover for the 2026-09-09 spark incident: a vLLM server started
with ``--reasoning-parser qwen3`` spent all 16384 output tokens inside
``reasoning_content`` (``content: null``, no tool calls) and Hermes reported
"Response remained truncated after 4 continuation attempts" — after making
ZERO continuation calls.
"""
from types import SimpleNamespace

import pytest

from agent import conversation_loop as cl
from agent import repetition_guard as rg


def _has_content_after_think(text):
    """Stand-in for ``agent._has_content_after_think_block``."""
    if "</think>" not in text:
        return False
    return bool(text.split("</think>", 1)[1].strip())


# ── 1. exhaustion detection ────────────────────────────────────────────


def test_reasoning_content_exhaustion_detected():
    """The incident shape: reasoning in a side channel, content=None."""
    assert cl._thinking_exhausted_signal(
        has_tool_calls=False,
        content=None,
        has_think_tags=False,
        has_content_after_think=_has_content_after_think,
        reasoning_text="现在让我检查第四步的表格内容。" * 200,
        reasoning_tokens=16384,
    ) is True


def test_usage_reasoning_tokens_alone_detect_exhaustion():
    """Providers that do not echo reasoning text still report it in usage."""
    assert cl._thinking_exhausted_signal(
        has_tool_calls=False,
        content="",
        has_think_tags=False,
        has_content_after_think=_has_content_after_think,
        reasoning_text="",
        reasoning_tokens=16384,
    ) is True


def test_think_tag_shape_still_detected():
    """Regression: the original <think>-tag form must keep working."""
    assert cl._thinking_exhausted_signal(
        has_tool_calls=False,
        content="<think>pondering</think>",
        has_think_tags=True,
        has_content_after_think=_has_content_after_think,
        reasoning_text="",
        reasoning_tokens=0,
    ) is True
    # Visible text after the think block is an ordinary truncation.
    assert cl._thinking_exhausted_signal(
        has_tool_calls=False,
        content="<think>pondering</think>Here is the answer so far",
        has_think_tags=True,
        has_content_after_think=_has_content_after_think,
        reasoning_text="",
        reasoning_tokens=0,
    ) is False


def test_visible_content_or_tool_calls_are_not_exhaustion():
    common = dict(
        has_think_tags=False,
        has_content_after_think=_has_content_after_think,
        reasoning_text="thinking",
        reasoning_tokens=4096,
    )
    assert cl._thinking_exhausted_signal(
        has_tool_calls=False, content="partial visible answer", **common
    ) is False
    assert cl._thinking_exhausted_signal(
        has_tool_calls=True, content=None, **common
    ) is False


def test_no_reasoning_evidence_is_not_exhaustion():
    """An empty truncation with no reasoning evidence stays continuable."""
    assert cl._thinking_exhausted_signal(
        has_tool_calls=False,
        content=None,
        has_think_tags=False,
        has_content_after_think=_has_content_after_think,
        reasoning_text="",
        reasoning_tokens=0,
    ) is False


# ── 2. reasoning extraction from the response ──────────────────────────


def test_reasoning_tokens_from_usage_object_and_dict():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            completion_tokens=16384,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=16384),
        )
    )
    assert cl._reasoning_tokens_from_usage(response) == 16384
    dict_usage = SimpleNamespace(
        usage={"completion_tokens_details": {"reasoning_tokens": 512}}
    )
    assert cl._reasoning_tokens_from_usage(dict_usage) == 512
    assert cl._reasoning_tokens_from_usage(SimpleNamespace()) == 0
    assert cl._reasoning_tokens_from_usage(SimpleNamespace(usage=None)) == 0


@pytest.mark.parametrize(
    "field", ["reasoning_content", "reasoning", "reasoning_details"]
)
def test_response_reasoning_text_reads_every_side_channel(field):
    message = SimpleNamespace(content=None, **{field: "deep thoughts"})
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)])
    assert cl._response_reasoning_text(response) == "deep thoughts"
    assert cl._response_reasoning_text(None, message) == "deep thoughts"


def test_response_reasoning_text_empty_when_absent():
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))]
    )
    assert cl._response_reasoning_text(response) == ""


# ── 3. attempt ceiling + give-up wording ───────────────────────────────


def test_max_attempts_default_and_env(monkeypatch):
    monkeypatch.delenv("HERMES_LENGTH_CONTINUATION_ATTEMPTS", raising=False)
    assert cl._length_continuation_max_attempts() == 3
    assert cl.LENGTH_CONTINUATION_MAX_ATTEMPTS == 3
    monkeypatch.setenv("HERMES_LENGTH_CONTINUATION_ATTEMPTS", "2")
    assert cl._length_continuation_max_attempts() == 2
    monkeypatch.setenv("HERMES_LENGTH_CONTINUATION_ATTEMPTS", "junk")
    assert cl._length_continuation_max_attempts() == 3


def test_giveup_message_reflects_actual_attempts():
    zero = cl._length_giveup_message(0, 3)
    assert "本轮输出过长且无法续写" in zero
    assert "续写 0 次" not in zero
    assert "3" not in zero.split("已保存")[0]

    twice = cl._length_giveup_message(2, 3)
    assert "已尝试续写 2 次" in twice

    for message in (zero, twice):
        assert "本轮没有写入任何内容" in message
        assert "已保存的草稿不受影响" in message
        assert "拆小" in message


def test_thinking_exhausted_message_is_chinese_and_quotes_budget():
    message = cl._thinking_exhausted_message(16384)
    assert "16384 tokens" in message
    assert "全部被模型的思考过程用完" in message
    assert "已保存的草稿不受影响" in message
    assert "本轮的输出上限" in cl._thinking_exhausted_message(None)


def test_repetition_and_tool_call_messages_are_chinese():
    assert "重复" in cl._repetition_abort_message()
    stall = cl._truncated_tool_call_message(True)
    length = cl._truncated_tool_call_message(False)
    assert "工具没有被执行" in stall
    assert "输出长度上限" in length
    for message in (stall, length):
        assert "已保存的草稿不受影响" in message


# ── 4. streaming repetition guard ──────────────────────────────────────

_LOOP_FRAGMENT = (
    "现在让我检查第四步的表格内容。我发现表格中只有广州、深圳、北京三个城市的信息，"
    "缺少上海这一行。让我构造一个精确的替换操作，把上海这一行加进去。"
)


def test_tail_repetition_detects_degenerate_loop():
    assert rg.tail_repetition_detected(_LOOP_FRAGMENT * 12) is True


def test_tail_repetition_ignores_ordinary_text():
    prose = " ".join(
        f"Step {i}: verify the DNS record for host-{i} and record the result."
        for i in range(60)
    )
    assert rg.tail_repetition_detected(prose) is False
    assert rg.tail_repetition_detected("short") is False
    assert rg.tail_repetition_detected(None) is False
    assert rg.tail_repetition_detected(12345) is False


def test_repetition_guard_env_switch(monkeypatch):
    monkeypatch.delenv("HERMES_REPETITION_GUARD", raising=False)
    assert rg.stream_repetition_guard_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("HERMES_REPETITION_GUARD", off)
        assert rg.stream_repetition_guard_enabled() is False
    monkeypatch.setenv("HERMES_REPETITION_GUARD", "1")
    assert rg.stream_repetition_guard_enabled() is True


# ── 5. reasoning_max_tokens is sent to the provider ────────────────────


def _agent_stub(**kwargs):
    return SimpleNamespace(request_overrides={}, **kwargs)


def test_reasoning_max_tokens_sets_thinking_token_budget():
    from agent.agent_init import _apply_reasoning_max_tokens

    agent = _agent_stub()
    assert _apply_reasoning_max_tokens(agent, {"reasoning_max_tokens": 2500}) == 2500
    assert agent.request_overrides["extra_body"]["thinking_token_budget"] == 2500


def test_reasoning_max_tokens_fills_openrouter_reasoning_object():
    from agent.agent_init import _apply_reasoning_max_tokens

    agent = _agent_stub()
    agent.request_overrides = {
        "extra_body": {"reasoning": {"enabled": True, "effort": "medium"}}
    }
    _apply_reasoning_max_tokens(agent, {"reasoning_max_tokens": 2500})
    extra_body = agent.request_overrides["extra_body"]
    assert extra_body["reasoning"] == {
        "enabled": True,
        "effort": "medium",
        "max_tokens": 2500,
    }
    assert extra_body["thinking_token_budget"] == 2500


def test_reasoning_max_tokens_absent_or_invalid_sends_nothing():
    from agent.agent_init import _apply_reasoning_max_tokens

    for cfg in ({}, {"reasoning_max_tokens": None}, {"reasoning_max_tokens": "abc"},
                {"reasoning_max_tokens": 0}, {"reasoning_max_tokens": True}):
        agent = _agent_stub()
        assert _apply_reasoning_max_tokens(agent, cfg) is None
        assert "extra_body" not in agent.request_overrides


def test_explicit_caller_budget_wins():
    from agent.agent_init import _apply_reasoning_max_tokens

    agent = _agent_stub()
    agent.request_overrides = {"extra_body": {"thinking_token_budget": 999}}
    _apply_reasoning_max_tokens(agent, {"reasoning_max_tokens": 2500})
    assert agent.request_overrides["extra_body"]["thinking_token_budget"] == 999


# ── 6. truncation-path logging ─────────────────────────────────────────


def test_length_truncation_logs_one_diagnostic_line(caplog):
    import logging

    caplog.set_level(logging.WARNING, logger="agent.conversation_loop")
    cl.log_length_truncation(
        session_id="20260909_102842_1f2a4f",
        usage=SimpleNamespace(prompt_tokens=16288, completion_tokens=16384),
        reasoning_tokens=16384,
        has_content=False,
        has_tool_calls=False,
        attempt=0,
        max_attempts=4,
        action="thinking_exhausted",
    )
    line = caplog.text
    assert "session=20260909_102842_1f2a4f" in line
    assert "finish_reason=length" in line
    assert "prompt=16288" in line
    assert "completion=16384" in line
    assert "reasoning=16384" in line
    assert "has_content=False" in line
    assert "has_tool_calls=False" in line
    assert "attempt=0/4" in line
    assert "action=thinking_exhausted" in line
