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


# ── 5. normalized-line loop guard (2026-09-10 WeCom degeneration) ──────

# The shape the model actually degenerated into: one short acknowledgement,
# re-quoted and re-framed on every pass. Byte-exact probes miss it because the
# framing changes; normalization collapses the variants.
_CAODI_LOOP_BLOCK = (
    '"收到。随时待命。"\n'
    "I'll output \"收到。随时待命。\"\n"
    "It's fine.\n"
    'Wait, "随时待命" is a bit "military".\n'
    '"收到。随时待命。"\n'
    "I'll output it.\n"
    "It's fine.\n"
    'Wait, "随时待命" is a bit "dramatic".\n'
)


def test_normalized_line_loop_trips_early_on_the_wecom_degeneration():
    """The verbatim tail probe needed 17152 characters on this text; the
    normalized-line probe must catch it inside 4000."""
    detector = rg.NormalizedLineLoopDetector()
    consumed = 0
    for _ in range(40):
        chunk = _CAODI_LOOP_BLOCK
        consumed += len(chunk)
        if detector.feed(chunk):
            break
    assert detector.tripped is True
    assert consumed <= 4000, f"tripped only after {consumed} characters"


def test_normalized_line_loop_survives_arbitrary_delta_boundaries():
    """Streaming splits lines anywhere; the detector buffers the partial tail."""
    text = _CAODI_LOOP_BLOCK * 6
    detector = rg.NormalizedLineLoopDetector()
    tripped = any(detector.feed(text[i : i + 7]) for i in range(0, len(text), 7))
    assert tripped is True


def test_normalized_line_loop_ignores_a_structured_list():
    """A real answer repeats STRUCTURE, not content — normalized lines stay distinct."""
    rows = "\n".join(
        f"| {host} | {ip} | {state} |"
        for host, ip, state in [
            ("bkm-node-a", "10.10.24.11", "running"), ("bkm-node-b", "10.10.24.12", "running"),
            ("bkm-node-c", "10.10.24.13", "stopped"), ("bkm-node-d", "10.10.24.14", "running"),
            ("gse-proxy-a", "10.10.25.21", "running"), ("gse-proxy-b", "10.10.25.22", "running"),
            ("job-exec-a", "10.10.26.31", "running"), ("job-exec-b", "10.10.26.32", "degraded"),
        ]
    )
    assert rg.normalized_line_loop_detected(rows + "\n") is False


def test_normalized_line_loop_ignores_a_numbered_runbook():
    steps = "\n".join(
        f"{n}. {action}" for n, action in enumerate(
            ["摘除负载均衡", "停止采集器", "备份配置目录", "升级 rpm 包", "回填配置",
             "启动采集器", "确认心跳恢复", "重新挂回负载均衡", "观察 10 分钟指标",
             "关闭变更单"], 1)
    )
    assert rg.normalized_line_loop_detected(steps + "\n") is False


def test_normalized_line_loop_ignores_short_repeated_markers():
    """Fence markers and bare list bullets recur legitimately."""
    text = "".join(f"```\nsystemctl status svc-{i}\n```\n" for i in range(6)) + ("- 是\n" * 6)
    assert rg.normalized_line_loop_detected(text) is False


def test_normalize_line_collapses_punctuation_and_digits():
    assert rg.normalize_line('  "收到。随时待命。" ') == rg.normalize_line("收到，随时待命")
    assert rg.normalize_line("Step 1: restart") == rg.normalize_line("step 1 restart!!")
    # Digits are part of the identity: a numbered checklist must stay distinct.
    assert rg.normalize_line("第 1 项检查通过") != rg.normalize_line("第 2 项检查通过")


_REPEATED_LINE = (
    "这是一句被模型反复输出的中文句子，用来验证复读守卫仍然按四次重复来判定，"
    "而不是三次；它本身足够长，所以三次重复就已经越过了两百字符的下限。"
)


def test_normalized_line_loop_needs_four_occurrences():
    three = _REPEATED_LINE + "\n" + (_REPEATED_LINE + "\n") * 2
    # Long past the 200-character floor, so only the repeat count can decide.
    assert len(three) >= rg.LINE_LOOP_MIN_TOTAL_CHARS
    assert rg.normalized_line_loop_detected(three) is False
    assert rg.normalized_line_loop_detected(three + _REPEATED_LINE + "\n") is True


def test_short_replies_never_trip_the_line_loop_guard():
    """2026-09-11 regression: 「你是谁？」 was aborted as a repetition loop.

    The real reply was a one-sentence identity statement the model restated a
    couple of times; under 200 characters nothing may be judged a loop.
    """
    identity = (
        "我是 haro管理员，由 嘉为科技 Haro 平台 提供。\n"
        "我是 haro管理员，由 嘉为科技 Haro 平台 提供。\n"
        "我是 haro管理员，由 嘉为科技 Haro 平台 提供。\n"
    )
    assert len(identity) < rg.LINE_LOOP_MIN_TOTAL_CHARS
    assert rg.normalized_line_loop_detected(identity) is False
    assert rg.tail_repetition_detected(identity) is False
    # …and a bare four-times-repeated short line is still under the floor.
    assert rg.normalized_line_loop_detected("同一句话。\n" * 4) is False


def test_short_normalized_lines_are_ignored():
    """A 5-rune line repeating is below LINE_LOOP_MIN_CHARS even when long."""
    text = "收到，好的。\n" * 60
    assert len(text) >= rg.LINE_LOOP_MIN_TOTAL_CHARS
    assert len(rg.normalize_line("收到，好的。")) < rg.LINE_LOOP_MIN_CHARS
    assert rg.normalized_line_loop_detected(text) is False


def test_normalized_line_loop_window_forgets_old_lines():
    """Three hits spread beyond the 40-line window must not accumulate."""
    filler = "\n".join(f"第 {i} 项检查通过" for i in range(1, 39))
    text = "\n".join([("重复的一行" + "\n" + filler) * 4])
    assert rg.normalized_line_loop_detected(text) is False
