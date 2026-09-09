"""Startup model-discovery / health probes must authenticate like chat does.

Covers:
  hermes_cli/runtime_provider.py — resolve_probe_api_key(), _auto_detect_local_model()
  hermes_cli/web_server.py       — GET /api/model/info context-length probe

A probe that omits ``Authorization`` gets a 401/403 from any keyed endpoint,
so model discovery silently returns nothing and the agent falls back to a
default model / context window while the chat request that follows succeeds.
"""

from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import runtime_provider as rp


class _Resp:
    ok = True

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


_ONE_MODEL = {"data": [{"id": "auto"}]}


def _model_config(**overrides):
    cfg = {
        "model": {
            "provider": "custom",
            "base_url": "http://10.10.42.35:8080/v1",
            "api_key": "dummy",
            "default": "auto",
        }
    }
    cfg["model"].update(overrides)
    return cfg


def _custom_provider_config(api_key="dummy", key_env=None):
    entry = {
        "name": "spark",
        "base_url": "http://10.10.42.35:8080/v1",
        "api_key": api_key,
    }
    if key_env:
        entry["key_env"] = key_env
    return {"custom_providers": [entry]}


# --------------------------------------------------------------------------
# resolve_probe_api_key(): the shared resolver both probe families reuse
# --------------------------------------------------------------------------


def test_resolve_probe_api_key_from_model_block(monkeypatch):
    monkeypatch.setattr(rp, "load_config", _model_config)
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == "dummy"


def test_resolve_probe_api_key_from_custom_provider(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: _custom_provider_config())
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == "dummy"


def test_resolve_probe_api_key_resolves_custom_provider_key_env(monkeypatch):
    monkeypatch.setenv("SPARK_KEY", "from-env")
    monkeypatch.setattr(
        rp, "load_config", lambda: _custom_provider_config(api_key="", key_env="SPARK_KEY")
    )
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == "from-env"


def test_resolve_probe_api_key_empty_when_unconfigured(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: {"model": {"default": "auto"}})
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == ""


def test_resolve_probe_api_key_does_not_leak_to_other_endpoint(monkeypatch):
    monkeypatch.setattr(rp, "load_config", _model_config)
    assert rp.resolve_probe_api_key("http://other.example:9000/v1") == ""


# --------------------------------------------------------------------------
# _auto_detect_local_model(): GET <base>/v1/models model discovery
# --------------------------------------------------------------------------


def _captured_headers(mock_get):
    assert mock_get.call_count == 1
    return mock_get.call_args.kwargs.get("headers")


def test_auto_detect_local_model_sends_bearer_from_model_block(monkeypatch):
    monkeypatch.setattr(rp, "load_config", _model_config)
    with patch("requests.get", return_value=_Resp(_ONE_MODEL)) as mock_get:
        assert rp._auto_detect_local_model("http://10.10.42.35:8080/v1") == "auto"
    assert _captured_headers(mock_get) == {"Authorization": "Bearer dummy"}


def test_auto_detect_local_model_sends_bearer_from_custom_provider(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: _custom_provider_config())
    with patch("requests.get", return_value=_Resp(_ONE_MODEL)) as mock_get:
        assert rp._auto_detect_local_model("http://10.10.42.35:8080/v1") == "auto"
    assert _captured_headers(mock_get) == {"Authorization": "Bearer dummy"}


def test_auto_detect_local_model_honours_explicit_api_key(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: {})
    with patch("requests.get", return_value=_Resp(_ONE_MODEL)) as mock_get:
        rp._auto_detect_local_model("http://10.10.42.35:8080/v1", api_key="explicit")
    assert _captured_headers(mock_get) == {"Authorization": "Bearer explicit"}


def test_auto_detect_local_model_sends_no_header_without_key(monkeypatch):
    monkeypatch.setattr(rp, "load_config", lambda: {"model": {"base_url": "http://x/v1"}})
    with patch("requests.get", return_value=_Resp(_ONE_MODEL)) as mock_get:
        rp._auto_detect_local_model("http://10.10.42.35:8080/v1")
    assert _captured_headers(mock_get) == {}


# --------------------------------------------------------------------------
# web dashboard: /api/model/info resolves the context window via a /models probe
# --------------------------------------------------------------------------


def test_model_info_context_probe_carries_api_key(monkeypatch):
    web_server = pytest.importorskip("hermes_cli.web_server")

    monkeypatch.setattr(web_server, "load_config", _model_config)
    monkeypatch.setattr(rp, "load_config", _model_config)

    ctx_probe = MagicMock(return_value=262144)
    with patch("agent.model_metadata.get_model_context_length", ctx_probe):
        web_server.get_model_info()

    assert ctx_probe.call_args.kwargs["api_key"] == "dummy"
