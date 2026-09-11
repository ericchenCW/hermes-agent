"""Unit tests for the truncated-response repetition guard (issue #86581)."""

from __future__ import annotations

from agent.repetition_guard import MIN_FRAGMENT_LENGTH, is_repetition_dominated

# The exact sentence from the #86581 incident (echoed hundreds of times by
# the model before the provider cut it off at finish_reason=length).
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"


class TestRepetitionGuard:
    def test_incident_shape_flags_repetition(self):
        # Narration + the echoed sentence on its own line, repeated (line path).
        text = ("We need to verify the model setting.\n" + _INCIDENT_ECHO + "\n") * 800
        assert is_repetition_dominated(text) is True

    def test_repeated_sentence_without_line_breaks_flags(self):
        # Repetition loop with no line breaks — exercises the window path.
        text = _INCIDENT_ECHO * 2000
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is True

    def test_long_legitimate_text_not_flagged(self):
        # Long, unique prose — no 60-char window ever repeats.
        text = " ".join(
            f"Sentence number {i} describes a distinct topic with unique words "
            f"such as quasar-{i} and nebula-{i} to keep every window distinct."
            for i in range(1200)
        )
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is False

    def test_short_fragment_never_flagged(self):
        # Below MIN_FRAGMENT_LENGTH the guard fails open — short truncations
        # are legitimately continued even if they look repetitive.
        assert is_repetition_dominated("A. " * 50) is False
        assert is_repetition_dominated("hello ") is False

    def test_repeat_not_dominant_not_flagged(self):
        # A repeated sentence scattered through a long unique text: repeated
        # windows exist but cover far less than half of the fragment.
        filler = " ".join(f"unique filler token {i}" for i in range(3000))
        text = filler + ("\n" + _INCIDENT_ECHO + "\n") * 30
        assert is_repetition_dominated(text) is False

    def test_non_string_inputs(self):
        assert is_repetition_dominated("") is False
        assert is_repetition_dominated(None) is False
        assert is_repetition_dominated(12345) is False


# ── streaming verbatim-tail probe: the 200-character floor ─────────────
# 2026-09-11 regression, item 12: 「写一段 50 字的英文自我介绍」 was aborted by the
# verbatim tail probe.  The floor that already governs the normalized-line probe
# now governs this one too — and, critically, it is measured on the WHOLE reply,
# not on the pre-sliced trailing window the streaming caller hands over (that
# slice reaches 2048 characters long before the floor could ever bind).

from agent import repetition_guard as rg  # noqa: E402

# A 50-word English self-introduction of the kind the model was asked for.
# It legitimately repeats a few phrases ("I help", "I can") — ordinary prose.
_SHORT_SELF_INTRO = (
    "Hi, I am an IT support assistant. I help colleagues with VPN access, "
    "mailbox setup and printer问题 every day. I can look things up in the "
    "knowledge base, I can walk you through the steps, and I can escalate to "
    "a human on duty when I cannot solve it myself."
)

# The 2026-09-09 CaoDi-style degeneration: one paragraph, verbatim, over and
# over inside the reasoning stream.
_DEGENERATE_PARAGRAPH = (
    "现在让我检查第四步的表格内容。我发现表格中只有广州、深圳、北京三个城市的信息，"
    "缺少上海这一行。让我构造一个精确的替换操作，把上海这一行加进去。"
)


class TestTailProbeFloor:
    def test_a_short_english_self_introduction_never_trips(self):
        assert len(_SHORT_SELF_INTRO) < rg.STREAM_MIN_REPLY_CHARS * 2
        assert rg.tail_repetition_detected(_SHORT_SELF_INTRO) is False

    def test_nothing_under_the_floor_is_ever_judged(self):
        # Even a blatant verbatim loop stays unjudged while the reply is short.
        loop = "abcdefghij" * 15  # 150 chars
        assert len(loop) < rg.STREAM_MIN_REPLY_CHARS
        assert rg.tail_repetition_detected(loop, fragment=10, min_repeats=3) is False

    def test_the_floor_is_measured_on_the_whole_reply_not_the_slice(self):
        """The streaming caller passes only the tail; ``total_chars`` carries the floor."""
        tail = _DEGENERATE_PARAGRAPH * 12
        assert rg.tail_repetition_detected(tail, total_chars=len(tail)) is True
        assert rg.tail_repetition_detected(tail, total_chars=199) is False
        assert rg.tail_repetition_detected(tail, total_chars=rg.STREAM_MIN_REPLY_CHARS) is True

    def test_the_degenerate_paragraph_still_trips_well_under_4000_chars(self):
        text = _DEGENERATE_PARAGRAPH * 12
        assert len(text) <= 4000
        assert rg.tail_repetition_detected(text) is True


# ── reasoning-loop retry (master verdict B, 2026-09-11) ────────────────
# A loop in the REASONING stream aborted the whole turn: four consecutive WeCom
# aborts in one session on 2026-09-11, and a stable 3/3 abort on 「写 50 字英文
# 自我介绍」.  The reasoning loop is now answered by cancelling the stream and
# re-asking ONCE with thinking switched off; the CONTENT stream still aborts.

import copy  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from tests.run_agent.test_partial_stream_finish_reason import _make_agent  # noqa: E402

# One paragraph the model repeats verbatim inside reasoning until the guard trips.
_LOOP = (
    "现在让我检查第四步的表格内容。我发现表格中只有广州、深圳、北京三个城市的信息，"
    "缺少上海这一行。让我构造一个精确的替换操作，把上海这一行加进去。\n"
)
_BODY = "Hi, I am an IT support assistant. I help colleagues with VPN and mailbox issues."
# Thinking knob the production route (Haro → bifrost → vLLM Qwen) actually sends.
_THINKING_ON = {"extra_body": {"thinking_token_budget": 2500}}


def _chunk(*, content=None, reasoning=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=None,
                            reasoning_content=reasoning, reasoning=None)
    return SimpleNamespace(choices=[SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)],
                           model=None, usage=None)


class _Stream:
    """Generator-ish stream that records whether it was abandoned mid-flight."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for c in self._chunks:
            self.consumed += 1
            yield c

    def close(self):
        self.closed = True

    @property
    def exhausted(self):
        return self.consumed >= len(self._chunks)


def _loop_stream():
    # 12 verbatim repeats: past STREAM_MIN_REPLY_CHARS and past both probes.
    # The trailing content chunk must NEVER be reached — the guard cancels first.
    return _Stream([_chunk(reasoning=_LOOP) for _ in range(12)]
                   + [_chunk(content="unreachable", finish_reason="stop")])


def _body_stream():
    return _Stream([_chunk(content=_BODY), _chunk(content="", finish_reason="stop")])


def _run(streams, monkeypatch, api_kwargs=_THINKING_ON):
    """Drive one streaming call; ``streams`` is consumed one per provider request."""
    calls: list[dict] = []
    made: list[_Stream] = []
    pending = list(streams)

    def _create(*_a, **kw):
        calls.append(kw)
        stream = pending.pop(0)
        made.append(stream)
        return stream

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _create
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    with patch("run_agent.AIAgent._create_request_openai_client", return_value=mock_client), \
         patch("run_agent.AIAgent._close_request_openai_client"):
        agent = _make_agent()
        seen: list[str] = []
        agent.stream_delta_callback = seen.append
        response = agent._interruptible_streaming_api_call(copy.deepcopy(api_kwargs))
    return response, seen, calls, made, agent


class TestReasoningLoopRetry:
    def test_reasoning_loop_cancels_the_stream_and_retries_without_thinking(self, monkeypatch):
        response, seen, calls, made, agent = _run([_loop_stream(), _body_stream()], monkeypatch)
        # Two provider requests: the aborted one and the thinking-off retry.
        assert len(calls) == 2
        # The first stream was cancelled, not drained: the trailing content chunk never ran.
        assert made[0].exhausted is False
        assert made[0].closed is True
        # The retry turned the request's own thinking knob off.
        retry_extra = calls[1]["extra_body"]
        assert retry_extra["thinking_token_budget"] == 0
        assert retry_extra["chat_template_kwargs"]["enable_thinking"] is False
        # …and the first request was NOT mutated in place.
        assert calls[0]["extra_body"]["thinking_token_budget"] == 2500
        # The body is delivered — no abort marker, no repetition notice.
        assert response.choices[0].message.content == _BODY
        assert response.choices[0].finish_reason == "stop"
        assert getattr(response, "_repetition_aborted", False) is False
        assert "".join(seen) == _BODY
        # The burnt attempt is charged as an API call.
        assert agent.session_api_calls == 1

    def test_a_second_reasoning_loop_falls_back_to_the_abort(self, monkeypatch):
        response, _seen, calls, _made, _agent = _run([_loop_stream(), _loop_stream()], monkeypatch)
        assert len(calls) == 2  # exactly one retry, never two
        assert getattr(response, "_repetition_aborted", None) == "reasoning"
        assert response.choices[0].finish_reason == "length"

    def test_a_content_loop_still_aborts_without_retrying(self, monkeypatch):
        content_loop = _Stream([_chunk(content=_LOOP) for _ in range(12)]
                               + [_chunk(content="tail", finish_reason="stop")])
        response, _seen, calls, _made, _agent = _run([content_loop], monkeypatch)
        assert len(calls) == 1  # the content stream is never re-asked
        assert getattr(response, "_repetition_aborted", None) == "content"
        assert response.choices[0].finish_reason == "length"

    def test_switch_off_restores_the_old_abort(self, monkeypatch):
        monkeypatch.setenv("HERMES_REPETITION_REASONING_RETRY", "0")
        response, _seen, calls, _made, _agent = _run([_loop_stream()], monkeypatch)
        assert len(calls) == 1
        assert getattr(response, "_repetition_aborted", None) == "reasoning"
        assert response.choices[0].finish_reason == "length"

    def test_no_thinking_switch_on_the_wire_falls_back_to_the_abort(self, monkeypatch):
        # Nothing safe to flip → the historical behaviour, not a 400-prone guess.
        response, _seen, calls, _made, _agent = _run([_loop_stream()], monkeypatch, api_kwargs={})
        assert len(calls) == 1
        assert getattr(response, "_repetition_aborted", None) == "reasoning"


class TestApplyThinkingOff:
    @pytest.mark.parametrize("kwargs, expected", [
        ({"extra_body": {"enable_thinking": True}}, {"enable_thinking": False}),
        ({"extra_body": {"think": True}}, {"think": False}),
        ({"extra_body": {"thinking": {"type": "enabled", "budget_tokens": 4000}}},
         {"thinking": {"type": "disabled"}}),
        ({"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}},
         {"chat_template_kwargs": {"enable_thinking": False}}),
    ])
    def test_known_switches_are_turned_off(self, kwargs, expected):
        out, switches = rg.apply_thinking_off(kwargs)
        assert out["extra_body"] == expected
        assert switches

    def test_openrouter_reasoning_object_is_disabled(self):
        out, _ = rg.apply_thinking_off({"extra_body": {"reasoning": {"effort": "high", "max_tokens": 4000}}})
        assert out["extra_body"]["reasoning"] == {"effort": "none", "enabled": False}

    def test_reasoning_effort_is_lowered_not_noned(self):
        # "none" is rejected by some routes; "low" is accepted wherever the field is.
        out, _ = rg.apply_thinking_off({"reasoning_effort": "high"})
        assert out["reasoning_effort"] == "low"

    def test_no_known_switch_returns_none(self):
        assert rg.apply_thinking_off({"model": "x", "messages": []}) is None
        assert rg.apply_thinking_off({"reasoning_effort": "low"}) is None
        assert rg.apply_thinking_off(None) is None

    @pytest.mark.parametrize("value, expected", [
        # bifrost drops thinking_token_budget / chat_template_kwargs; `user` is the
        # standard field that survives to thinkcap, so this is the real switch.
        ("haro;bot=b8ba5f7a;think=budget:6000", "haro;bot=b8ba5f7a;think=off"),
        ("haro;bot=b8ba5f7a;think=inherit", "haro;bot=b8ba5f7a;think=off"),
        ("haro;bot=b8ba5f7a", "haro;bot=b8ba5f7a;think=off"),
    ])
    def test_haro_user_field_is_switched_to_think_off(self, value, expected):
        out, switches = rg.apply_thinking_off({"extra_body": {"user": value}})
        assert out["extra_body"]["user"] == expected
        assert "user.think=off" in switches

    @pytest.mark.parametrize("value, expected", [
        ("haro;bot=b8ba5f7a;think=budget:6000", "haro;bot=b8ba5f7a;think=off"),
        ("haro;bot=b8ba5f7a;think=inherit", "haro;bot=b8ba5f7a;think=off"),
        ("haro;bot=b8ba5f7a", "haro;bot=b8ba5f7a;think=off"),
    ])
    def test_top_level_user_field_is_switched_too(self, value, expected):
        out, switches = rg.apply_thinking_off({"user": value})
        assert out["user"] == expected
        assert switches == ["user.think=off"]

    def test_non_haro_user_field_is_left_alone(self):
        # Another deployment's `user` is an opaque identifier — rewriting it would
        # corrupt whatever routing/quota it carries.
        assert rg.apply_thinking_off({"user": "u-12345"}) is None
        assert rg.apply_thinking_off({"extra_body": {"user": "u-12345"}}) is None

    def test_user_already_off_is_not_counted_as_a_switch(self):
        assert rg.apply_thinking_off({"user": "haro;bot=b8ba5f7a;think=off"}) is None

    def test_the_env_switch_gates_the_retry(self, monkeypatch):
        assert rg.reasoning_retry_enabled() is True
        for off in ("0", "false", "no", "off"):
            monkeypatch.setenv("HERMES_REPETITION_REASONING_RETRY", off)
            assert rg.reasoning_retry_enabled() is False
        monkeypatch.setenv("HERMES_REPETITION_REASONING_RETRY", "1")
        assert rg.reasoning_retry_enabled() is True
