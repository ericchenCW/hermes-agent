"""HERMES_GATEWAY_DISABLED_COMMANDS hides listed slash commands from every user."""
from types import SimpleNamespace

from gateway.run import GatewayRunner


def _check(cmd):
    runner = SimpleNamespace(config=None)
    source = SimpleNamespace(user_id="u1", platform=None)
    return GatewayRunner._check_slash_access(runner, source, cmd)


def test_disabled_commands_env(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_DISABLED_COMMANDS", "help, /Commands")
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    assert _check("help") == "⛔ /help is not available here."
    assert _check("commands") == "⛔ /commands is not available here."


def test_disabled_commands_env_zh(monkeypatch):
    monkeypatch.setenv("HERMES_GATEWAY_DISABLED_COMMANDS", "help")
    monkeypatch.setenv("HERMES_LANGUAGE", "zh")
    assert _check("help") == "⛔ 这里未开放 /help 命令。"


def test_disabled_commands_env_unset_falls_through(monkeypatch):
    monkeypatch.delenv("HERMES_GATEWAY_DISABLED_COMMANDS", raising=False)
    # policy_for_source(None, ...) is not reached for disabled commands; for
    # an unset env the gate must fall through to the regular policy path.
    import gateway.run as run_mod
    monkeypatch.setattr(
        "gateway.slash_access.policy_for_source",
        lambda cfg, src: SimpleNamespace(enabled=False, can_run=lambda u, c: True, user_allowed_commands=set()),
    )
    assert _check("help") is None
