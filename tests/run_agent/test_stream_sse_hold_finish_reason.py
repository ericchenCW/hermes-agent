"""Regression: the SSE-leak hold must not swallow finish_reason / usage that
ride on the same chunk as held text, and must not trigger mid-line.

Field case (WeCom bot, 2026-09-03): a reply ending in ``MEDIA:/opt/.../x.jpg``
streamed as deltas ``"MEDIA"``, ``":/opt/..."``, ``"9.jpg"`` — the last delta
carried ``finish_reason="stop"``.  The ``":/opt"`` delta looked like an SSE
comment line, the hold branch ``continue``d past finish_reason capture, and a
clean end-of-stream was classified as a mid-stream drop → full retry.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.run_agent.test_partial_stream_finish_reason import _make_agent, _make_stream_chunk


def _run(chunks, monkeypatch):
    def _stream():
        for c in chunks:
            yield c
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = lambda *a, **kw: _stream()
    with patch("run_agent.AIAgent._create_request_openai_client", return_value=mock_client), \
         patch("run_agent.AIAgent._close_request_openai_client"):
        agent = _make_agent()
        seen = []
        agent.stream_delta_callback = seen.append
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
        response = agent._interruptible_streaming_api_call({})
    return response, seen


def test_finish_reason_on_held_chunk_is_not_a_stream_drop(monkeypatch):
    chunks = [
        _make_stream_chunk(content="步骤见下图。\n\nMEDIA"),
        _make_stream_chunk(content=":/opt/data/cache/sheets/steps-178842415"),
        _make_stream_chunk(content="9.jpg", finish_reason="stop"),
    ]
    response, seen = _run(chunks, monkeypatch)
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == "步骤见下图。\n\nMEDIA:/opt/data/cache/sheets/steps-1788424159.jpg"
    assert "".join(seen) == "步骤见下图。\n\nMEDIA:/opt/data/cache/sheets/steps-1788424159.jpg"


def test_colon_delta_mid_line_is_streamed_immediately(monkeypatch):
    chunks = [
        _make_stream_chunk(content="访问 https"),
        _make_stream_chunk(content="://km.cwoa.net/login"),
        _make_stream_chunk(content=" 登录。", finish_reason="stop"),
    ]
    response, seen = _run(chunks, monkeypatch)
    assert response.choices[0].finish_reason == "stop"
    # the ":" delta is ordinary text mid-line: it is fired as a delta, not held
    assert seen[:2] == ["访问 https", "://km.cwoa.net/login"]
