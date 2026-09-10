"""Replay cover for the reasoning-leak reply guard (``agent/leak_guard.py``).

The two fixtures below are verbatim openings of real leaks pulled from the
spark bifrost transcript during the 2026-09-10 forensics:

  * ``CAODI_CONTENT_OPENING`` — session ``20260910_082108_c2e419aa``,
    ``finish_reason=length``, ``reasoning_tokens=2499`` against a thinkcap
    budget of 2500.  The reasoning tail ended on "… Wait, the prompt" and
    ``content`` opened with ``says:`` plus the model's own compressed
    restatement of its system-prompt style rules, which the WeCom adapter
    then pushed to an external user.
  * ``BKMONITOR_CONTENT_OPENING`` — 2026-09-09, ``finish_reason=stop``, same
    shape with no continuation: ``content`` opened mid-question with a
    dangling ``)`` and carried on thinking in the open.

They are kept literal on purpose: a paraphrase would not exercise the rule
table against what the model actually emitted.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent import leak_guard as lg


# ── fixtures ───────────────────────────────────────────────────────────

CAODI_CONTENT_OPENING = (
    'says:\n'
    '"Be direct... No filler... No restating... Plain claims... Agree because it\'s right..."\n'
    '\n'
    'User: "好的" (Okay).\n'
    'Assistant: "收到。随时待命。"\n'
    '\n'
    'This is a valid exchange.\n'
    '\n'
    "Let's check if the user asked to update a skill in the *first* message of the conversation?\n"
    '"保存到技能/记忆" (Save to skill/memory).\n'
)

CAODI_REASONING_TAIL = (
    '\nWait, is there any possibility that "好的" is a command?\nNo.\n'
    '\nI\'ll reply "收到。随时待命。" to be polite but brief.\n'
    '\nSo just "收到。"\n\nWait, the prompt\n'
)

BKMONITOR_CONTENT_OPENING = (
    ') if bkmonitor is working?"\n'
    'This sounds like they want to verify the service.\n'
    '\n'
    'I will provide the `bkcec` command for the platform status.\n'
    'And `systemctl` for agent status.\n'
    '\n'
    'No specific KB file needed.\n'
    'No tool calls needed.\n'
    '\n'
    'Wait, user said "这个表示 bkmonitor 安装正常... 那我怎么在主机上用命令查看".\n'
)

NORMAL_CHINESE_REPLY = (
    "在主机上确认 bkmonitor 采集是否正常，可以按下面三步走：\n\n"
    "1. 查看进程：`ps -ef | grep bkmonitorbeat`，正常应能看到常驻进程。\n"
    "2. 查看状态：`/usr/local/gse/gseagent/bin/gsectl status`。\n"
    "3. 查看日志：`tail -n 100 /var/log/gse/bkmonitorbeat.log`，无 ERROR 即正常。\n\n"
    "如果第 1 步就没有进程，说明插件没起来，需要在节点管理里重新下发一次。\n"
) * 6

ENGLISH_DOC_REPLY = (
    "Let me draft the runbook section you asked for.\n\n"
    "## Restarting the collector\n\n"
    "1. Drain the node from the load balancer.\n"
    "2. Restart the agent with `systemctl restart bkmonitorbeat`.\n"
    "3. Confirm the heartbeat returns within 60 seconds.\n"
) * 6


def _usage(reasoning_tokens: int, completion_tokens: int = 16384):
    """A provider usage object shaped like the vLLM / OpenAI-compatible one."""
    return SimpleNamespace(
        completion_tokens=completion_tokens,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
    )


def _chunks(text: str, size: int = 200):
    return [text[i : i + size] for i in range(0, len(text), size)]


@pytest.fixture()
def audit_log(tmp_path, monkeypatch):
    """Point the guard's audit sink at a temp HERMES_HOME and read it back."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_THINK_BUDGET_HINT", raising=False)

    def _lines():
        path = tmp_path / "logs" / "replyguard.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return _lines


def _replay(content: str, *, budget, reasoning_tokens, chunk_size: int = 200):
    """Drive a full stream: content deltas, then the trailing usage chunk.

    Returns ``(guard, emitted_text)`` where ``emitted_text`` is exactly what the
    adapter would have pushed to the user.
    """
    guard = lg.StreamLeakGuard(budget=budget, platform="wecom", session="s-test", subject="CaoDi")
    emitted = []
    for chunk in _chunks(content, chunk_size):
        emitted.append(guard.on_content_delta(chunk).emit)
    emitted.append(guard.on_usage(_usage(reasoning_tokens)).emit)
    emitted.append(guard.finish().emit)
    return guard, "".join(emitted)


# ── 1. the CaoDi leak (2026-09-10) ─────────────────────────────────────


def test_caodi_leak_is_convicted_and_nothing_reaches_the_user(audit_log):
    """The incident replay: 2499/2500 reasoning tokens plus a mid-thought opening."""
    guard, emitted = _replay(CAODI_CONTENT_OPENING * 30, budget=2500, reasoning_tokens=2499)

    assert guard.convicted is True
    # R2 is the necessary condition; R1 (opens with "says:") and R4 ("\nUser: ")
    # are the shape evidence.
    assert guard.rules[0] == "R2"
    assert "R1" in guard.rules and "R4" in guard.rules
    # Nothing at all left the guard: the hold ran to the verdict.
    assert emitted == ""
    assert "Be direct" not in emitted
    assert guard.final_text("whatever the model wrote") == lg.REDACTION_TEXT

    events = audit_log()
    assert len(events) == 1
    assert events[0]["event"] == "reply.redacted"
    assert events[0]["rule"].startswith("R2+")
    assert "R1" in events[0]["rule"]
    assert events[0]["reasoning_tokens"] == 2499
    assert events[0]["budget"] == 2500
    assert events[0]["original_len"] == len(CAODI_CONTENT_OPENING * 30)
    # The audit record must never carry the leaked body.
    assert "Be direct" not in json.dumps(events[0], ensure_ascii=False)


def test_reasoning_tail_and_content_head_form_one_sentence():
    """Documents the mechanism: the cap cut "Wait, the prompt" / "says:" in half."""
    assert CAODI_REASONING_TAIL.rstrip().endswith("Wait, the prompt")
    assert CAODI_CONTENT_OPENING.startswith("says:")


# ── 2. the 09-09 leak (finish_reason=stop, no continuation) ────────────


def test_bkmonitor_leak_is_convicted(audit_log):
    guard, emitted = _replay(BKMONITOR_CONTENT_OPENING * 30, budget=2500, reasoning_tokens=2500)

    assert guard.convicted is True
    assert "R1" in guard.rules  # dangling ")" opener
    assert "R3" in guard.rules  # "Wait, user said" at a line start
    assert emitted == ""
    assert len(audit_log()) == 1


# ── 3. a normal answer must be untouched ───────────────────────────────


def test_normal_chinese_reply_streams_through_unheld(audit_log):
    guard, emitted = _replay(NORMAL_CHINESE_REPLY, budget=2500, reasoning_tokens=800)

    assert guard.convicted is False
    assert emitted == NORMAL_CHINESE_REPLY  # every delta went out live
    assert guard.final_text(NORMAL_CHINESE_REPLY) == NORMAL_CHINESE_REPLY
    assert audit_log() == []


# ── 4. R3 alone never convicts ─────────────────────────────────────────


def test_english_document_opening_with_let_me_is_released(audit_log):
    """A user asking for an English runbook gets a reply opening "Let me …".

    R1 and R3 both fire, but the reasoning budget was barely touched, so R2
    acquits and the buffered text is flushed in full.
    """
    guard, emitted = _replay(ENGLISH_DOC_REPLY, budget=2500, reasoning_tokens=120)

    assert guard.convicted is False
    assert emitted == ENGLISH_DOC_REPLY
    assert audit_log() == []


# ── 5. R2 alone is audit-only ──────────────────────────────────────────


def test_budget_exhausted_but_well_formed_reply_is_only_audited(audit_log):
    """A reply that used its whole thinking budget and still answered properly
    must reach the user; the operator only gets a log line."""
    guard, emitted = _replay(NORMAL_CHINESE_REPLY, budget=2500, reasoning_tokens=2499)

    assert guard.convicted is False
    assert emitted == NORMAL_CHINESE_REPLY
    events = audit_log()
    assert len(events) == 1
    assert events[0]["event"] == "reply.audited"
    assert events[0]["rule"] == "R2"
    assert events[0]["redacted"] is False


# ── 6. no budget knowledge → no conviction ─────────────────────────────


def test_without_a_budget_the_leak_text_is_not_convicted(audit_log):
    """R2 is a necessary condition, so an unknown budget can only fail open —
    and the guard must not even hold, since no evidence can ever arrive."""
    guard, emitted = _replay(CAODI_CONTENT_OPENING * 3, budget=None, reasoning_tokens=2499)

    assert guard.convicted is False
    assert emitted == CAODI_CONTENT_OPENING * 3
    assert audit_log() == []


# ── rule table unit cover ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, rule",
    [
        ("says: the thing", "R1"),
        (") if bkmonitor is working?", "R1"),
        ("means the agent is healthy", "R1"),
        ("Fine.\nWait, the user actually asked something else.", "R3"),
        ("ok\nThe system prompt says: be terse", "R3"),
        ("done\nI should just reply with the command", "R3"),
        ("x\nFinal Decision: restart the agent", "R3"),
        ("x\nThinking Process: enumerate the options", "R3"),
        ("blah\nUser: 好的\nAssistant: 收到", "R4"),
        ("recap of Turn 3: the user asked about snmp", "R4"),
    ],
)
def test_shape_rules(text, rule):
    assert rule in lg.evaluate_prefix(text)


def test_shape_rules_ignore_a_plain_chinese_answer():
    assert lg.evaluate_prefix("你好，我可以帮你查看 bkmonitor 的状态。") == []


def test_shape_rules_only_look_at_the_opening():
    """A leak announces itself immediately; a 40k-char answer is not rescanned."""
    text = "正常回答。\n" * 200 + "\nUser: 好的\n"
    assert lg.evaluate_prefix(text) == []


@pytest.mark.parametrize(
    "tokens, budget, expected",
    [
        (2499, 2500, True),   # vLLM stops one short of the cap
        (2500, 2500, True),
        (2496, 2500, True),   # inside BUDGET_SLACK
        (2495, 2500, False),
        (800, 2500, False),
        (None, 2500, False),
        (2499, None, False),
        (0, 2500, False),
    ],
)
def test_reasoning_exhausted(tokens, budget, expected):
    assert lg.reasoning_exhausted(tokens, budget) is expected


def test_reasoning_tokens_read_from_dict_and_object_usage():
    assert lg.reasoning_tokens_from_usage(_usage(2499)) == 2499
    assert lg.reasoning_tokens_from_usage({"completion_tokens_details": {"reasoning_tokens": 7}}) == 7
    assert lg.reasoning_tokens_from_usage(None) is None
    assert lg.reasoning_tokens_from_usage(SimpleNamespace()) is None


# ── budget resolution ──────────────────────────────────────────────────


def test_budget_comes_from_request_overrides(monkeypatch):
    monkeypatch.delenv("HERMES_THINK_BUDGET_HINT", raising=False)
    agent = SimpleNamespace(request_overrides={"extra_body": {"thinking_token_budget": 2500}})
    assert lg.resolve_thinking_budget(agent) == 2500


def test_budget_falls_back_to_the_openrouter_shape(monkeypatch):
    monkeypatch.delenv("HERMES_THINK_BUDGET_HINT", raising=False)
    agent = SimpleNamespace(request_overrides={"extra_body": {"reasoning": {"max_tokens": 1800}}})
    assert lg.resolve_thinking_budget(agent) == 1800


def test_budget_falls_back_to_the_env_hint(monkeypatch):
    """thinkcap owns the cap in the spark deployment; Hermes never sends it."""
    monkeypatch.setenv("HERMES_THINK_BUDGET_HINT", "2500")
    assert lg.resolve_thinking_budget(SimpleNamespace(request_overrides={})) == 2500


def test_budget_is_none_when_nothing_declares_one(monkeypatch):
    monkeypatch.delenv("HERMES_THINK_BUDGET_HINT", raising=False)
    assert lg.resolve_thinking_budget(SimpleNamespace()) is None


def test_guard_is_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HERMES_REPLY_LEAK_GUARD", "0")
    assert lg.make_stream_leak_guard(SimpleNamespace()) is None
    monkeypatch.setenv("HERMES_REPLY_LEAK_GUARD", "1")
    assert lg.make_stream_leak_guard(SimpleNamespace()) is not None


# ── usage arriving mid-stream convicts immediately ─────────────────────


def test_usage_before_the_last_chunk_convicts_without_waiting(audit_log):
    """Relays that report usage early must not have to wait for the stream end."""
    guard = lg.StreamLeakGuard(budget=2500, platform="wecom", session="s", subject="u")
    emitted = guard.on_content_delta(CAODI_CONTENT_OPENING[:200]).emit
    assert emitted == ""  # held on suspicion
    assert guard.on_usage(_usage(2499)).convicted is True
    assert guard.on_content_delta("more leaked reasoning").emit == ""
    assert guard.finish().emit == ""
    assert guard.convicted is True


# ── integration: the streaming emit path honours the guard ─────────────


def test_streaming_emit_path_is_gated_by_the_guard():
    """``_StreamingCall._emit_text`` must hold, then drop, then replace.

    Built without a provider: only the emit funnel and the guard interact here,
    which is exactly the seam the WeCom adapter streams out of.
    """
    from agent.chat_completion_helpers import _StreamingCall

    sent: list[str] = []
    agent = SimpleNamespace(
        _fire_stream_delta=sent.append, platform="wecom", session_id="s", chat_id="CaoDi",
        request_overrides={"extra_body": {"thinking_token_budget": 2500}},
    )
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = agent
    call.deltas_were_sent = {"yes": False}
    call.first_delta_fired = {"done": True}
    call.on_first_delta = None
    call._leak_guard = lg.make_stream_leak_guard(agent)

    call._emit_text(CAODI_CONTENT_OPENING[:200])
    assert sent == []  # held on suspicion, nothing on the wire

    call._leak_guard.on_usage(_usage(2499))
    call._emit_text("still leaking")
    assert sent == []
    assert call._leak_guard.final_text(None) == lg.REDACTION_TEXT


def test_streaming_emit_path_passes_a_normal_reply_through():
    from agent.chat_completion_helpers import _StreamingCall

    sent: list[str] = []
    agent = SimpleNamespace(
        _fire_stream_delta=sent.append, platform="wecom", session_id="s", chat_id="CaoDi",
        request_overrides={"extra_body": {"thinking_token_budget": 2500}},
    )
    call = _StreamingCall.__new__(_StreamingCall)
    call.agent = agent
    call.deltas_were_sent = {"yes": False}
    call.first_delta_fired = {"done": True}
    call.on_first_delta = None
    call._leak_guard = lg.make_stream_leak_guard(agent)

    call._emit_text("你好，我来帮你查 bkmonitor。")
    assert sent == ["你好，我来帮你查 bkmonitor。"]
    assert call.deltas_were_sent["yes"] is True
