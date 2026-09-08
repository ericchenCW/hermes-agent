"""The MCP circuit breaker must only count *transport* failures.

Field case: a business-level rejection from an MCP tool (e.g. a document
server answering "draft find is not unique") is returned as a normal
``tools/call`` result carrying ``isError`` / an ``error`` field.  The old
handler bumped the breaker for any result JSON containing an ``"error"``
key, so three such refusals in a row opened the breaker and locked out
*every other tool* on that server for the 60 s cooldown.

Contract pinned here:

* tool-level ``isError`` results never count, and never trip the breaker;
* an ``error`` field inside an otherwise successful result payload never
  counts either;
* transport failures (connection errors, timeouts, JSON-RPC protocol
  errors) still count and still trip the breaker;
* a successful round-trip resets the count to zero.
"""
import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip("mcp.client.auth.oauth2")

from tools import mcp_tool_loop as _mcp_loop  # noqa: E402
from tests.tools.test_mcp_circuit_breaker import _cleanup, _install_stub_server  # noqa: E402


def _ok_result(text="ok", structured=None):
    result = MagicMock()
    result.is_error = False
    block = MagicMock()
    block.text = text
    result.content = [block]
    result.structured_content = structured
    result.meta = None
    return result


def _tool_error_result(text="draft find is not unique"):
    result = MagicMock()
    result.is_error = True
    block = MagicMock()
    block.text = text
    result.content = [block]
    result.structured_content = None
    result.meta = None
    return result


def test_tool_level_errors_do_not_count_or_trip_breaker(monkeypatch, tmp_path):
    """Three consecutive business-level refusals: no strikes, no breaker."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    calls = {"n": 0}

    async def _call_tool(*a, **kw):
        calls["n"] += 1
        return _tool_error_result()

    _install_stub_server(mcp_tool, "srv", _call_tool)
    _mcp_loop._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "draft_find", 10.0)
        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
            parsed = json.loads(handler({}))
            # The refusal still reaches the model as a tool error…
            assert "error" in parsed, parsed
            assert "not unique" in parsed["error"]
            # …but it is not a connectivity signal.
            assert mcp_tool._server_error_counts.get("srv", 0) == 0

        # Every call actually hit the session — nothing was short-circuited.
        assert calls["n"] == mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        assert "srv" not in mcp_tool._server_breaker_opened_at

        # An unrelated tool on the same server is still usable.
        parsed = json.loads(handler({}))
        assert "unreachable" not in json.dumps(parsed).lower()
    finally:
        _cleanup(mcp_tool, "srv")


def test_error_field_in_successful_payload_does_not_count(monkeypatch, tmp_path):
    """A server that puts an ``error`` key in structuredContent is still a
    healthy transport."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    async def _call_tool(*a, **kw):
        return _ok_result(text="", structured={"error": "validation failed"})

    _install_stub_server(mcp_tool, "srv", _call_tool)
    _mcp_loop._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "t", 10.0)
        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 2):
            handler({})
            assert mcp_tool._server_error_counts.get("srv", 0) == 0
        assert "srv" not in mcp_tool._server_breaker_opened_at
    finally:
        _cleanup(mcp_tool, "srv")


def test_transport_failures_still_trip_the_breaker(monkeypatch, tmp_path):
    """Connection-level failures keep their strikes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    async def _call_tool(*a, **kw):
        raise ConnectionError("connection reset by peer")

    _install_stub_server(mcp_tool, "srv", _call_tool)
    _mcp_loop._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "t", 10.0)
        for n in range(1, mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 1):
            handler({})
            assert mcp_tool._server_error_counts.get("srv", 0) == n

        assert "srv" in mcp_tool._server_breaker_opened_at
        # Breaker is open: the next call short-circuits.
        parsed = json.loads(handler({}))
        assert "unreachable" in parsed.get("error", "").lower(), parsed
    finally:
        _cleanup(mcp_tool, "srv")


def test_successful_round_trip_resets_the_count(monkeypatch, tmp_path):
    """Transport strikes below threshold are cleared by one good call."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    mode = {"fail": True}

    async def _call_tool(*a, **kw):
        if mode["fail"]:
            raise TimeoutError("read timed out")
        return _ok_result()

    _install_stub_server(mcp_tool, "srv", _call_tool)
    _mcp_loop._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "t", 10.0)
        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD - 1):
            handler({})
        assert mcp_tool._server_error_counts["srv"] == (
            mcp_tool._CIRCUIT_BREAKER_THRESHOLD - 1
        )

        mode["fail"] = False
        parsed = json.loads(handler({}))
        assert parsed.get("result") == "ok", parsed
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
        assert "srv" not in mcp_tool._server_breaker_opened_at
    finally:
        _cleanup(mcp_tool, "srv")


def test_tool_level_error_after_transport_strikes_resets_them(monkeypatch, tmp_path):
    """A tool-level refusal is a proven round-trip, so it clears strikes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool
    from tools.mcp_tool_handlers import _make_tool_handler

    mode = {"fail": True}

    async def _call_tool(*a, **kw):
        if mode["fail"]:
            raise ConnectionError("boom")
        return _tool_error_result()

    _install_stub_server(mcp_tool, "srv", _call_tool)
    _mcp_loop._ensure_mcp_loop()

    try:
        handler = _make_tool_handler("srv", "t", 10.0)
        handler({})
        handler({})
        assert mcp_tool._server_error_counts["srv"] == 2

        mode["fail"] = False
        handler({})
        assert mcp_tool._server_error_counts.get("srv", 0) == 0
    finally:
        _cleanup(mcp_tool, "srv")
