"""Content deltas that arrive after a tool_calls delta must still travel
the normal ``_fire_stream_delta`` path.

Field case (qwen3.6 through the gateway): once a response had emitted any
``tool_calls`` delta, the chat-completions streaming loop routed later
content deltas straight into ``agent.stream_delta_callback``.  That is the
CLI callback only — it bypassed ``_fire_stream_delta``, and with it
``agent._stream_callback`` (the gateway's ``message.delta`` source), the
think/context scrubbers and the plugin stream hooks.  The gateway saw
``thinking.delta`` frames and a final ``message.complete``, but not a
single ``message.delta``.
"""
from unittest.mock import MagicMock, patch

import pytest

from tests.run_agent.test_partial_stream_finish_reason import (
    _make_agent,
    _make_stream_chunk,
    _make_tool_call_delta,
)


def _run(chunks, monkeypatch, wire):
    def _stream():
        for c in chunks:
            yield c

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stream()
    with patch("run_agent.AIAgent._create_request_openai_client", return_value=mock_client), \
         patch("run_agent.AIAgent._close_request_openai_client"):
        agent = _make_agent()
        wire(agent)
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})
    return response


def _tool_then_text_chunks():
    return [
        _make_stream_chunk(
            tool_calls=[_make_tool_call_delta(
                index=0, tc_id="call_1", name="search", arguments="{}",
            )],
        ),
        _make_stream_chunk(content="Let me look that up for you."),
        _make_stream_chunk(content="  Checking now.", finish_reason="tool_calls"),
    ]


def test_gateway_callback_receives_deltas_after_tool_calls(monkeypatch):
    """``_stream_callback`` (gateway message.delta) must not be skipped."""
    gateway_seen = []
    response = _run(
        _tool_then_text_chunks(),
        monkeypatch,
        lambda agent: setattr(agent, "_stream_callback", gateway_seen.append),
    )

    assert response.choices[0].finish_reason == "tool_calls"
    assert "".join(gateway_seen) == "Let me look that up for you.  Checking now."


def test_no_double_send_to_the_cli_callback(monkeypatch):
    """Both consumers get the text exactly once each."""
    cli_seen = []
    gateway_seen = []

    def _wire(agent):
        agent.stream_delta_callback = cli_seen.append
        agent._stream_callback = gateway_seen.append

    _run(_tool_then_text_chunks(), monkeypatch, _wire)

    expected = "Let me look that up for you.  Checking now."
    assert "".join(cli_seen) == expected
    assert "".join(gateway_seen) == expected


def test_think_tags_are_scrubbed_on_the_tool_call_path(monkeypatch):
    """Going through ``_fire_stream_delta`` also buys the reasoning
    scrubbing the direct-callback path never had."""
    gateway_seen = []
    chunks = [
        _make_stream_chunk(
            tool_calls=[_make_tool_call_delta(
                index=0, tc_id="call_1", name="search", arguments="{}",
            )],
        ),
        _make_stream_chunk(content="<think>internal</think>Visible."),
        _make_stream_chunk(content="", finish_reason="tool_calls"),
    ]
    _run(
        chunks,
        monkeypatch,
        lambda agent: setattr(agent, "_stream_callback", gateway_seen.append),
    )

    joined = "".join(gateway_seen)
    assert "internal" not in joined
    assert joined == "Visible."


def test_text_only_stream_is_unchanged(monkeypatch):
    """The no-tool-call path keeps its existing behaviour."""
    gateway_seen = []
    chunks = [
        _make_stream_chunk(content="Hello "),
        _make_stream_chunk(content="world.", finish_reason="stop"),
    ]
    response = _run(
        chunks,
        monkeypatch,
        lambda agent: setattr(agent, "_stream_callback", gateway_seen.append),
    )
    assert response.choices[0].finish_reason == "stop"
    assert "".join(gateway_seen) == "Hello world."
