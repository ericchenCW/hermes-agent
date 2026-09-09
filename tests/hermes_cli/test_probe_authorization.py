"""Startup model-discovery / health probes must authenticate like chat does.

Covers:
  hermes_cli/runtime_provider.py — resolve_probe_api_key(), _auto_detect_local_model()
  hermes_cli/web_server.py       — GET /api/model/info context-length probe
  agent/model_metadata.py        — expand_probe_api_key(), _auth_headers()

A probe that omits ``Authorization`` gets a 401/403 from any keyed endpoint,
so model discovery silently returns nothing and the agent falls back to a
default model / context window while the chat request that follows succeeds.
"""

from unittest.mock import MagicMock, patch

import pytest

from agent import model_metadata as mm
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


# --------------------------------------------------------------------------
# ${ENV} expansion: probes must present the same *expanded* credential the
# chat path does. Container deployments write ``api_key: ${HARO_MODEL_API_KEY}``
# into config.yaml, and a probe credential can reach the probe layer from a
# caller that never ran the expansion (raw-YAML gateway reads, key_env lookups,
# plugin pass-through) — the endpoint then sees ``Bearer ${HARO_MODEL_API_KEY}``
# and 401s while the chat request that follows authenticates fine.
# --------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_probe_warn_once():
    mm._warned_unresolved_probe_refs.clear()
    yield
    mm._warned_unresolved_probe_refs.clear()


def test_expand_probe_api_key_brace_ref(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    assert mm.expand_probe_api_key("${HARO_MODEL_API_KEY}") == "vk-real"
    assert mm._auth_headers("${HARO_MODEL_API_KEY}") == {"Authorization": "Bearer vk-real"}


def test_expand_probe_api_key_env_prefixed_ref(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    assert mm.expand_probe_api_key("${env:HARO_MODEL_API_KEY}") == "vk-real"


def test_expand_probe_api_key_leaves_bare_dollar_var_verbatim(monkeypatch):
    """``$VAR`` is not an env reference in config.yaml.

    ``hermes_cli.config._expand_env_vars`` (the chat path) only resolves
    ``${VAR}`` / ``${env:VAR}``. Expanding the bare shape here would make the
    probe authenticate with a different credential than the chat request that
    follows it — the exact divergence this module exists to prevent.
    """
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    assert mm.expand_probe_api_key("$HARO_MODEL_API_KEY") == "$HARO_MODEL_API_KEY"


def test_expand_probe_api_key_is_idempotent_for_plain_keys(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    assert mm.expand_probe_api_key("sk-plain-123") == "sk-plain-123"
    assert mm.expand_probe_api_key(mm.expand_probe_api_key("${HARO_MODEL_API_KEY}")) == "vk-real"
    assert mm.expand_probe_api_key("") == ""


def test_expand_probe_api_key_unset_env_drops_header_and_warns(monkeypatch, caplog):
    monkeypatch.delenv("HARO_MODEL_API_KEY", raising=False)
    with caplog.at_level("WARNING", logger="agent.model_metadata"):
        assert mm.expand_probe_api_key("${HARO_MODEL_API_KEY}") == ""
        # Probes also run periodically — the warning must not repeat per call.
        assert mm.expand_probe_api_key("${HARO_MODEL_API_KEY}") == ""
    warnings = [r for r in caplog.records if "HARO_MODEL_API_KEY" in r.getMessage()]
    assert len(warnings) == 1
    assert mm._auth_headers("${HARO_MODEL_API_KEY}") == {}


def test_fetch_endpoint_model_metadata_expands_placeholder(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()

    captured = {}

    class _MetaResp:
        status_code = 200
        headers: dict = {}

        def json(self):
            return {"data": [{"id": "auto"}]}

        def raise_for_status(self):
            pass

        def close(self):
            pass

    def _fake_get(url, **kwargs):
        captured.setdefault("headers", kwargs.get("headers"))
        return _MetaResp()

    with patch("requests.get", _fake_get):
        mm.fetch_endpoint_model_metadata(
            "http://10.10.42.35:8080/v1",
            api_key="${HARO_MODEL_API_KEY}",
            force_refresh=True,
        )
    assert captured["headers"] == {"Authorization": "Bearer vk-real"}


# --------------------------------------------------------------------------
# resolve_probe_api_key(): all three credential sources go through expansion
# --------------------------------------------------------------------------


def test_resolve_probe_api_key_expands_model_block_placeholder(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    monkeypatch.setattr(
        rp, "load_config", lambda: _model_config(api_key="${HARO_MODEL_API_KEY}")
    )
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == "vk-real"


def test_resolve_probe_api_key_expands_custom_provider_placeholder(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: _custom_provider_config(api_key="${HARO_MODEL_API_KEY}"),
    )
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == "vk-real"


def test_resolve_probe_api_key_expands_key_env_placeholder(monkeypatch):
    """``key_env`` reads os.environ directly — a var whose *value* is itself a
    template (chained container env) must still resolve."""
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    monkeypatch.setenv("SPARK_KEY", "${HARO_MODEL_API_KEY}")
    monkeypatch.setattr(
        rp,
        "load_config",
        lambda: _custom_provider_config(api_key="", key_env="SPARK_KEY"),
    )
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == "vk-real"


def test_resolve_probe_api_key_unset_placeholder_is_empty(monkeypatch):
    monkeypatch.delenv("HARO_MODEL_API_KEY", raising=False)
    monkeypatch.setattr(
        rp, "load_config", lambda: _model_config(api_key="${HARO_MODEL_API_KEY}")
    )
    assert rp.resolve_probe_api_key("http://10.10.42.35:8080/v1") == ""


def test_auto_detect_local_model_expands_explicit_placeholder(monkeypatch):
    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    monkeypatch.setattr(rp, "load_config", lambda: {})
    with patch("requests.get", return_value=_Resp(_ONE_MODEL)) as mock_get:
        rp._auto_detect_local_model(
            "http://10.10.42.35:8080/v1", api_key="${HARO_MODEL_API_KEY}"
        )
    assert _captured_headers(mock_get) == {"Authorization": "Bearer vk-real"}


# --------------------------------------------------------------------------
# gateway startup probe, end to end: the header that actually leaves the box
# --------------------------------------------------------------------------


def test_gateway_startup_probe_sends_expanded_bearer(monkeypatch, tmp_path):
    """``_resolve_gateway_model_context`` is the gateway's startup/banner probe.

    It reads config.yaml as raw YAML, so the ``api_key`` it carries can still
    be a ``${VAR}`` template; assert on the outgoing Authorization header
    rather than on the intermediate value.
    """
    gateway_run = pytest.importorskip("gateway.run")

    monkeypatch.setenv("HARO_MODEL_API_KEY", "vk-real")
    (tmp_path / "config.yaml").write_text(
        "model:\n"
        "  provider: custom\n"
        "  default: auto\n"
        "  base_url: http://10.10.42.35:8080/v1\n"
        "  api_key: ${HARO_MODEL_API_KEY}\n"
        "custom_providers:\n"
        "  - name: bifrost\n"
        "    base_url: http://10.10.42.35:8080/v1\n"
        "    api_key: ${HARO_MODEL_API_KEY}\n"
        "    models: [auto]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(gateway_run, "_hermes_home", lambda: tmp_path, raising=False)
    # Stand in for the credential shape a raw-YAML caller produces: the gateway
    # reads config.yaml with yaml.safe_load (``_load_gateway_config``), so an
    # api_key that reaches the probe layer can still be the env template.
    monkeypatch.setattr(
        gateway_run,
        "_resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "custom",
            "base_url": "http://10.10.42.35:8080/v1",
            "api_key": "${HARO_MODEL_API_KEY}",
        },
    )

    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()

    seen = []

    class _ProbeResp:
        status_code = 200
        ok = True
        headers: dict = {}
        text = "{}"

        def json(self):
            return {"data": [{"id": "auto", "context_length": 262144}]}

        def raise_for_status(self):
            pass

        def close(self):
            pass

    def _fake_get(url, **kwargs):
        seen.append(dict(kwargs.get("headers") or {}))
        return _ProbeResp()

    with patch("requests.get", _fake_get):
        gateway_run._resolve_gateway_model_context("auto")

    assert seen, "gateway startup resolved no probe at all"
    assert all(h.get("Authorization") == "Bearer vk-real" for h in seen), seen
