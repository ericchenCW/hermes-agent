"""platform_send control-socket verb — outbound push for external schedulers.

Haro's scheduled jobs run outside the gateway process but must deliver their
result into a live IM session; only the process holding the platform
connection may send. The verb takes {platform, chat_id, text}, marshals the
adapter's ``send`` coroutine onto the gateway loop and answers synchronously.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from gateway.control_socket import (
    PLATFORM_SEND_ERR_ADAPTER_NOT_READY,
    PLATFORM_SEND_ERR_BAD_REQUEST,
    PLATFORM_SEND_ERR_PLATFORM_NOT_FOUND,
    PLATFORM_SEND_ERR_SEND_FAILED,
    PLATFORM_SEND_ERR_TIMEOUT,
    PLATFORM_SEND_MAX_TEXT_CHARS,
    PLATFORM_SEND_VERB,
    GatewayControlServer,
    build_platform_send_handler,
    platform_send,
)

SECRET = "机密正文：季度营收 1234 万"


class _FakeAdapter:
    """Minimal live-adapter stand-in: records sends, answers like SendResult."""

    def __init__(self, *, connected=True, message_id="msg-1", raises=None,
                 success=True, error=None, hang=False):
        self.is_connected = connected
        self._message_id = message_id
        self._raises = raises
        self._success = success
        self._error = error
        self._hang = hang
        self.sent: list[tuple[str, str]] = []

    async def send(self, chat_id, content, *args, **kwargs):
        self.sent.append((chat_id, content))
        if self._hang:
            await asyncio.Event().wait()  # never returns
        if self._raises is not None:
            raise self._raises
        return SimpleNamespace(
            success=self._success, message_id=self._message_id, error=self._error
        )


def _run_handler(adapter_map, request, *, timeout=None):
    """Drive the handler from a worker thread while a real loop runs here.

    Mirrors production: the handler executes on the control socket's executor
    thread and bridges back to the gateway's event loop.
    """

    async def scenario():
        loop = asyncio.get_running_loop()
        handler = build_platform_send_handler(
            adapter_map.get, loop=loop, timeout=timeout
        )
        return await loop.run_in_executor(None, handler, request)

    return asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------

def test_send_returns_ok_and_message_id():
    adapter = _FakeAdapter(message_id="msg-42")
    result = _run_handler(
        {"wecom": adapter},
        {"verb": PLATFORM_SEND_VERB, "platform": "wecom",
         "chat_id": "ericyu", "text": SECRET},
    )
    assert result == {"ok": True, "message_id": "msg-42"}
    assert adapter.sent == [("ericyu", SECRET)]


def test_message_id_is_null_when_adapter_returns_none():
    result = _run_handler(
        {"wecom": _FakeAdapter(message_id=None)},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert result == {"ok": True, "message_id": None}


# ---------------------------------------------------------------------------
# 2. bad_request
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "request_payload",
    [
        {"chat_id": "ericyu", "text": "hi"},                     # no platform
        {"platform": "wecom", "text": "hi"},                     # no chat_id
        {"platform": "wecom", "chat_id": "ericyu"},              # no text
        {"platform": "wecom", "chat_id": "ericyu", "text": ""},  # empty text
        {"platform": "wecom", "chat_id": "ericyu", "text": "   "},
        {"platform": "wecom", "chat_id": "ericyu", "text": 42},  # wrong type
        {"platform": "", "chat_id": "ericyu", "text": "hi"},
    ],
)
def test_bad_request_for_missing_or_empty_fields(request_payload):
    adapter = _FakeAdapter()
    result = _run_handler({"wecom": adapter}, request_payload)
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_BAD_REQUEST
    assert result["message"]
    assert adapter.sent == []  # never reached the adapter


def test_bad_request_when_text_too_long():
    adapter = _FakeAdapter()
    result = _run_handler(
        {"wecom": adapter},
        {"platform": "wecom", "chat_id": "ericyu",
         "text": "x" * (PLATFORM_SEND_MAX_TEXT_CHARS + 1)},
    )
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_BAD_REQUEST
    assert str(PLATFORM_SEND_MAX_TEXT_CHARS) in result["message"]
    assert adapter.sent == []


def test_max_length_text_is_accepted():
    result = _run_handler(
        {"wecom": _FakeAdapter()},
        {"platform": "wecom", "chat_id": "ericyu",
         "text": "x" * PLATFORM_SEND_MAX_TEXT_CHARS},
    )
    assert result["ok"] is True


# ---------------------------------------------------------------------------
# 3. platform_not_found / adapter_not_ready
# ---------------------------------------------------------------------------

def test_platform_not_found_when_gateway_does_not_serve_it():
    result = _run_handler(
        {"wecom": _FakeAdapter()},
        {"platform": "telegram", "chat_id": "123", "text": "hi"},
    )
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_PLATFORM_NOT_FOUND
    assert "telegram" in result["message"]


def test_adapter_not_ready_when_not_connected():
    adapter = _FakeAdapter(connected=False)
    result = _run_handler(
        {"wecom": adapter},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_ADAPTER_NOT_READY
    assert adapter.sent == []


# ---------------------------------------------------------------------------
# 4. send_failed
# ---------------------------------------------------------------------------

def test_send_failed_when_adapter_raises():
    result = _run_handler(
        {"wecom": _FakeAdapter(raises=RuntimeError("errcode 846604 req_id expired"))},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_SEND_FAILED
    assert "846604" in result["message"]


def test_send_failed_when_adapter_reports_failure():
    result = _run_handler(
        {"wecom": _FakeAdapter(success=False, error="errcode 846607 rate limited")},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_SEND_FAILED
    assert "846607" in result["message"]


# ---------------------------------------------------------------------------
# 5. timeout
# ---------------------------------------------------------------------------

def test_timeout_when_send_never_returns():
    result = _run_handler(
        {"wecom": _FakeAdapter(hang=True)},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
        timeout=0.2,
    )
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_TIMEOUT
    assert "0.2" in result["message"]


def test_timeout_budget_comes_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_PLATFORM_SEND_TIMEOUT", "0.15")
    result = _run_handler(
        {"wecom": _FakeAdapter(hang=True)},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert result["error"] == PLATFORM_SEND_ERR_TIMEOUT
    assert "0.15" in result["message"]


# ---------------------------------------------------------------------------
# 6. Logging never carries the message body
# ---------------------------------------------------------------------------

def test_log_records_metadata_but_not_the_text(caplog):
    with caplog.at_level(logging.INFO, logger="gateway.control_socket"):
        _run_handler(
            {"wecom": _FakeAdapter()},
            {"platform": "wecom", "chat_id": "ericyu", "text": SECRET},
        )
    lines = [r.getMessage() for r in caplog.records]
    joined = "\n".join(lines)
    assert SECRET not in joined
    assert "季度营收" not in joined
    assert any(
        "platform_send" in line
        and "wecom" in line
        and "ericyu" in line
        and f"text_len={len(SECRET)}" in line
        and "result=ok" in line
        for line in lines
    )


def test_failure_log_also_omits_the_text(caplog):
    with caplog.at_level(logging.INFO, logger="gateway.control_socket"):
        _run_handler(
            {"wecom": _FakeAdapter(raises=RuntimeError(SECRET))},
            {"platform": "wecom", "chat_id": "ericyu", "text": SECRET},
        )
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in joined
    assert f"result={PLATFORM_SEND_ERR_SEND_FAILED}" in joined


# ---------------------------------------------------------------------------
# Wire frame + full socket round trip
# ---------------------------------------------------------------------------

def test_response_frame_reports_outcome_at_top_level():
    """A failed send must not read as ok at the frame level."""

    def handler(request):
        from gateway.control_socket import _platform_send_failure

        return _platform_send_failure(PLATFORM_SEND_ERR_BAD_REQUEST, "nope")

    server = GatewayControlServer(
        home="/tmp", verb_handlers={PLATFORM_SEND_VERB: handler}
    )
    raw = json.dumps({"verb": PLATFORM_SEND_VERB, "id": 3}).encode()
    frame = json.loads(server.handle_request_line(raw).decode())
    assert frame["ok"] is False
    assert frame["error"] == PLATFORM_SEND_ERR_BAD_REQUEST
    assert frame["message"] == "nope"
    assert frame["result"] == {"ok": False, "error": PLATFORM_SEND_ERR_BAD_REQUEST,
                               "message": "nope"}
    assert frame["id"] == 3


def test_existing_zero_arg_verbs_keep_the_v1_envelope():
    server = GatewayControlServer(
        home="/tmp", verb_handlers={"pause-for-update": lambda: {"pausing": True}}
    )
    raw = json.dumps({"verb": "pause-for-update", "id": 1}).encode()
    frame = json.loads(server.handle_request_line(raw).decode())
    assert frame["ok"] is True
    assert frame["result"] == {"pausing": True}
    assert "pausing" not in frame  # not hoisted


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
def test_client_roundtrip_over_real_socket(tmp_path):
    """Full client → socket → executor → gateway loop → adapter → answer."""
    adapter = _FakeAdapter(message_id="msg-live")
    outcome: dict = {}

    async def scenario():
        loop = asyncio.get_running_loop()
        server = GatewayControlServer(
            home=tmp_path,
            verb_handlers={
                PLATFORM_SEND_VERB: build_platform_send_handler(
                    {"wecom": adapter}.get, loop=loop
                )
            },
        )
        assert await server.start()
        try:
            # The blocking client must not run on the loop it is waiting on.
            def _call():
                return platform_send(
                    tmp_path, "wecom", "ericyu", SECRET, timeout=5.0
                )

            outcome["result"] = await asyncio.get_running_loop().run_in_executor(
                None, _call
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    assert outcome["result"] == {"ok": True, "message_id": "msg-live"}
    assert adapter.sent == [("ericyu", SECRET)]


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
def test_client_reports_platform_not_found_when_no_gateway(tmp_path):
    result = platform_send(tmp_path, "wecom", "ericyu", "hi", timeout=0.5)
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_PLATFORM_NOT_FOUND


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
def test_client_reports_not_found_against_gateway_without_the_verb(tmp_path):
    """Back-compat: an older gateway answers 'unknown verb', not a crash."""

    async def scenario():
        server = GatewayControlServer(home=tmp_path)
        assert await server.start()
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, lambda: platform_send(tmp_path, "wecom", "e", "hi", timeout=2.0)
            )
        finally:
            await server.stop()

    result = asyncio.run(scenario())
    assert result["ok"] is False
    assert result["error"] == PLATFORM_SEND_ERR_PLATFORM_NOT_FOUND
