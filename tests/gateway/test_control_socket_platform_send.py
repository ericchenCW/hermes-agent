"""platform_send control-socket verb — outbound push for external schedulers.

Haro's scheduled jobs run outside the gateway process but must deliver their
result into a live IM session; only the process holding the platform
connection may send. The verb takes {platform, chat_id, text, request_id},
marshals the adapter's ``send`` coroutine onto the gateway loop and answers
synchronously.

Wire contract (agreed with the Haro side, 2026-09-10):

  success  {"ok":true,"protocol":1,"id":…,
            "result":{"message_id":…,"request_id":…,"platform":…,"chat_id":…}}
  failure  {"ok":false,"protocol":1,"id":…,"error":"<code>: <detail>"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from types import SimpleNamespace

import pytest

from gateway.control_socket import (
    PLATFORM_SEND_ERR_BAD_REQUEST,
    PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
    PLATFORM_SEND_ERR_RATE_LIMITED,
    PLATFORM_SEND_ERR_SEND_FAILED,
    PLATFORM_SEND_ERR_TIMEOUT,
    PLATFORM_SEND_MAX_TEXT_CHARS,
    PLATFORM_SEND_VERB,
    GatewayControlServer,
    VerbError,
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


def _run_handler_error(adapter_map, request, *, timeout=None) -> VerbError:
    """Run the handler expecting a failure, and return the raised VerbError."""
    with pytest.raises(VerbError) as excinfo:
        _run_handler(adapter_map, request, timeout=timeout)
    return excinfo.value


# ---------------------------------------------------------------------------
# 1. Happy path — the four-field result
# ---------------------------------------------------------------------------

def test_send_returns_the_contract_result():
    adapter = _FakeAdapter(message_id="msg-42")
    result = _run_handler(
        {"wecom": adapter},
        {"verb": PLATFORM_SEND_VERB, "platform": "wecom", "chat_id": "ericyu",
         "text": SECRET, "request_id": "req-7"},
    )
    assert result == {
        "message_id": "msg-42",
        "request_id": "req-7",
        "platform": "wecom",
        "chat_id": "ericyu",
    }
    assert adapter.sent == [("ericyu", SECRET)]


def test_message_id_is_empty_string_when_adapter_returns_none():
    result = _run_handler(
        {"wecom": _FakeAdapter(message_id=None)},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert result["message_id"] == ""


@pytest.mark.parametrize(
    "request_payload, expected",
    [
        ({"request_id": "uuid-abc"}, "uuid-abc"),   # echoed verbatim
        ({}, ""),                                    # absent → empty string
        ({"request_id": None}, ""),                  # null → empty string
        ({"request_id": 12345}, ""),                 # wrong type → empty, not a reject
        ({"request_id": "not a uuid at all!"}, "not a uuid at all!"),  # unvalidated
    ],
)
def test_request_id_is_echoed_without_validation(request_payload, expected):
    result = _run_handler(
        {"wecom": _FakeAdapter()},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi", **request_payload},
    )
    assert result["request_id"] == expected


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
        {"platform": 7, "chat_id": "ericyu", "text": "hi"},
    ],
)
def test_bad_request_for_missing_or_empty_fields(request_payload):
    adapter = _FakeAdapter()
    error = _run_handler_error({"wecom": adapter}, request_payload)
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
    assert error.detail
    assert adapter.sent == []  # never reached the adapter


@pytest.mark.parametrize("platform", ["telegram", "slack", "dingtalk"])
def test_bad_request_for_unsupported_platform(platform):
    """Only wecom is served; anything else is a caller mistake, not an outage."""
    adapter = _FakeAdapter()
    error = _run_handler_error(
        {"wecom": adapter},
        {"platform": platform, "chat_id": "123", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
    assert platform in error.detail
    assert adapter.sent == []


def test_bad_request_when_text_too_long():
    adapter = _FakeAdapter()
    error = _run_handler_error(
        {"wecom": adapter},
        {"platform": "wecom", "chat_id": "ericyu",
         "text": "x" * (PLATFORM_SEND_MAX_TEXT_CHARS + 1)},
    )
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
    assert str(PLATFORM_SEND_MAX_TEXT_CHARS) in error.detail
    assert adapter.sent == []


def test_max_length_text_is_accepted():
    result = _run_handler(
        {"wecom": _FakeAdapter()},
        {"platform": "wecom", "chat_id": "ericyu",
         "text": "x" * PLATFORM_SEND_MAX_TEXT_CHARS},
    )
    assert result["message_id"] == "msg-1"


# ---------------------------------------------------------------------------
# 3. platform_unavailable
# ---------------------------------------------------------------------------

def test_platform_unavailable_when_no_live_adapter():
    error = _run_handler_error(
        {},  # gateway serves nothing right now
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE
    assert "wecom" in error.detail


def test_platform_unavailable_when_adapter_not_connected():
    adapter = _FakeAdapter(connected=False)
    error = _run_handler_error(
        {"wecom": adapter},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE
    assert adapter.sent == []


# ---------------------------------------------------------------------------
# 4. send_failed / rate_limited
# ---------------------------------------------------------------------------

def test_send_failed_when_adapter_raises():
    error = _run_handler_error(
        {"wecom": _FakeAdapter(raises=RuntimeError("errcode 846604 req_id expired"))},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_SEND_FAILED
    assert "846604" in error.detail  # errcode + errmsg kept verbatim


def test_send_failed_when_adapter_reports_failure():
    error = _run_handler_error(
        {"wecom": _FakeAdapter(success=False, error="errcode 40003 invalid userid")},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_SEND_FAILED
    assert "40003" in error.detail
    assert "invalid userid" in error.detail


def test_rate_limited_when_send_result_carries_846607():
    error = _run_handler_error(
        {"wecom": _FakeAdapter(
            success=False, error="send failed: errcode 846607, errmsg: too many request"
        )},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_RATE_LIMITED
    assert "846607" in error.detail


def test_rate_limited_when_adapter_raises_with_846607():
    error = _run_handler_error(
        {"wecom": _FakeAdapter(
            raises=RuntimeError("aibot_send_msg errcode=846607 too many request")
        )},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_RATE_LIMITED


# ---------------------------------------------------------------------------
# 5. timeout
# ---------------------------------------------------------------------------

def test_timeout_when_send_never_returns():
    error = _run_handler_error(
        {"wecom": _FakeAdapter(hang=True)},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
        timeout=0.2,
    )
    assert error.code == PLATFORM_SEND_ERR_TIMEOUT
    assert "0.2" in error.detail


def test_timeout_budget_comes_from_env(monkeypatch):
    monkeypatch.setenv("HERMES_PLATFORM_SEND_TIMEOUT", "0.15")
    error = _run_handler_error(
        {"wecom": _FakeAdapter(hang=True)},
        {"platform": "wecom", "chat_id": "ericyu", "text": "hi"},
    )
    assert error.code == PLATFORM_SEND_ERR_TIMEOUT
    assert "0.15" in error.detail


def test_default_timeout_is_fifteen_seconds():
    from gateway.control_socket import PLATFORM_SEND_DEFAULT_TIMEOUT, _platform_send_timeout

    assert PLATFORM_SEND_DEFAULT_TIMEOUT == 15.0
    assert _platform_send_timeout() == 15.0


# ---------------------------------------------------------------------------
# 6. Logging never carries the message body
# ---------------------------------------------------------------------------

def test_log_records_metadata_but_not_the_text(caplog):
    with caplog.at_level(logging.INFO, logger="gateway.control_socket"):
        _run_handler(
            {"wecom": _FakeAdapter()},
            {"platform": "wecom", "chat_id": "ericyu", "text": SECRET,
             "request_id": "req-log"},
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
        and "request_id=req-log" in line
        and "result=ok" in line
        for line in lines
    )


def test_failure_log_also_omits_the_text(caplog):
    with caplog.at_level(logging.INFO, logger="gateway.control_socket"):
        _run_handler_error(
            {"wecom": _FakeAdapter(raises=RuntimeError(SECRET))},
            {"platform": "wecom", "chat_id": "ericyu", "text": SECRET},
        )
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET not in joined
    assert f"result={PLATFORM_SEND_ERR_SEND_FAILED}" in joined


# ---------------------------------------------------------------------------
# Wire frames
# ---------------------------------------------------------------------------

def _frame(server, request: dict) -> dict:
    return json.loads(server.handle_request_line(json.dumps(request).encode()).decode())


def test_success_frame_carries_result_and_echoes_id():
    adapter = _FakeAdapter(message_id="msg-live")

    async def scenario():
        loop = asyncio.get_running_loop()
        server = GatewayControlServer(
            home="/tmp",
            verb_handlers={
                PLATFORM_SEND_VERB: build_platform_send_handler(
                    {"wecom": adapter}.get, loop=loop
                )
            },
        )
        return await loop.run_in_executor(
            None,
            lambda: _frame(server, {
                "verb": PLATFORM_SEND_VERB, "id": "abc", "protocol": 1,
                "platform": "wecom", "chat_id": "ericyu", "text": "hi",
                "request_id": "req-9",
            }),
        )

    frame = asyncio.run(scenario())
    assert frame == {
        "ok": True,
        "protocol": 1,
        "id": "abc",
        "result": {
            "message_id": "msg-live",
            "request_id": "req-9",
            "platform": "wecom",
            "chat_id": "ericyu",
        },
    }
    # The old flattening is gone: nothing hoisted beside the envelope.
    assert "message_id" not in frame


def test_failure_frame_is_code_colon_detail_with_no_result():
    def handler(request):
        raise VerbError(PLATFORM_SEND_ERR_BAD_REQUEST, "text is required")

    server = GatewayControlServer(
        home="/tmp", verb_handlers={PLATFORM_SEND_VERB: handler}
    )
    frame = _frame(server, {"verb": PLATFORM_SEND_VERB, "id": 3})
    assert frame == {
        "ok": False,
        "protocol": 1,
        "id": 3,
        "error": f"{PLATFORM_SEND_ERR_BAD_REQUEST}: text is required",
    }
    assert "result" not in frame
    assert "message" not in frame


def test_unknown_verb_lists_platform_send_among_supported_verbs():
    async def scenario():
        loop = asyncio.get_running_loop()
        server = GatewayControlServer(
            home="/tmp",
            verb_handlers={
                "pause-for-update": lambda: {"pausing": True},
                PLATFORM_SEND_VERB: build_platform_send_handler(
                    {"wecom": _FakeAdapter()}.get, loop=loop
                ),
            },
        )
        return _frame(server, {"verb": "no-such-verb", "id": 1})

    frame = asyncio.run(scenario())
    assert frame["ok"] is False
    assert PLATFORM_SEND_VERB in frame["supported_verbs"]
    assert "pause-for-update" in frame["supported_verbs"]


def test_existing_zero_arg_verbs_keep_the_v1_envelope():
    server = GatewayControlServer(
        home="/tmp", verb_handlers={"pause-for-update": lambda: {"pausing": True}}
    )
    frame = _frame(server, {"verb": "pause-for-update", "id": 1})
    assert frame["ok"] is True
    assert frame["result"] == {"pausing": True}
    assert "pausing" not in frame  # not hoisted


# ---------------------------------------------------------------------------
# Full socket round trip (client ↔ server)
# ---------------------------------------------------------------------------

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
                    tmp_path, "wecom", "ericyu", SECRET,
                    request_id="req-rt", timeout=5.0,
                )

            outcome["result"] = await loop.run_in_executor(None, _call)
        finally:
            await server.stop()

    asyncio.run(scenario())
    assert outcome["result"] == {
        "ok": True,
        "result": {
            "message_id": "msg-live",
            "request_id": "req-rt",
            "platform": "wecom",
            "chat_id": "ericyu",
        },
    }
    assert adapter.sent == [("ericyu", SECRET)]


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
def test_client_roundtrip_surfaces_coded_failure(tmp_path):
    adapter = _FakeAdapter(success=False, error="errcode 846607, errmsg: too many")
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
            outcome["result"] = await loop.run_in_executor(
                None,
                lambda: platform_send(tmp_path, "wecom", "ericyu", "hi", timeout=5.0),
            )
        finally:
            await server.stop()

    asyncio.run(scenario())
    result = outcome["result"]
    assert result["ok"] is False
    assert result["error"].startswith(f"{PLATFORM_SEND_ERR_RATE_LIMITED}: ")
    assert "result" not in result


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
def test_client_reports_platform_unavailable_when_no_gateway(tmp_path):
    result = platform_send(tmp_path, "wecom", "ericyu", "hi", timeout=0.5)
    assert result["ok"] is False
    assert result["error"].startswith(f"{PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE}: ")


@pytest.mark.skipif(sys.platform == "win32", reason="unix socket transport")
def test_client_reports_unavailable_against_gateway_without_the_verb(tmp_path):
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
    assert result["error"].startswith(f"{PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE}: ")
    assert "unknown verb" in result["error"]
