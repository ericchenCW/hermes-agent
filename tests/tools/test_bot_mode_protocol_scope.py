"""Tests for ``agent.bot_mode_protocol_scope`` — the A2A reach switch.

Upstream, the Bot Mode teammate protocol (system-prompt section + the
``message_agent`` tool) exists ONLY in a profile's canonical "Bot Chat"
session. Hermes session titles are globally unique, so an external
orchestrator that drives one bot through many sessions can never have more
than one of them qualify. ``bot_mode_protocol_scope: all`` widens the gate
to every ordinary session of a Bot-Mode-managed profile.

What must hold:
  * default ("bot_chat") — byte-for-byte upstream behaviour;
  * "all" — any title qualifies, but ONLY on a managed install, and never
    for group rooms, cron agents or subagents;
  * an invalid config value falls back to "bot_chat" (fail safe).
"""

import json
import textwrap
from pathlib import Path

import pytest

from agent.agent_init import _resolve_bot_mode_protocol_scope
from tools import bot_mode_dm, bot_mode_probe


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


def _managed_home(tmp_path: Path, *, teammates=("researcher",)) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    for name in teammates:
        d = home / "profiles" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "profile.yaml").write_text(
            textwrap.dedent(
                """\
                description: teammate for tests
                ui_meta:
                  hermes-bots:
                    shape: cloud
                """
            ),
            encoding="utf-8",
        )
    return home


class _FakeDB:
    def __init__(self, home: Path, title: str):
        self.db_path = str(home / "state.db")
        self._title = title

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    def __init__(self, home: Path, title: str = "Bot Chat", scope: str = "bot_chat"):
        self._session_db = _FakeDB(home, title)
        self.session_id = "sess-1"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self._bot_mode_protocol_scope = scope
        self.tools: list = []
        self.valid_tool_names: set = set()


# ── config resolution ────────────────────────────────────────────────────────


def test_scope_default_and_valid_values():
    assert _resolve_bot_mode_protocol_scope("bot_chat") == "bot_chat"
    assert _resolve_bot_mode_protocol_scope("all") == "all"
    # Config files are hand-written: tolerate case/whitespace.
    assert _resolve_bot_mode_protocol_scope("  ALL  ") == "all"


@pytest.mark.parametrize("raw", ["", None, "everything", "Bot Chat", 7, True, []])
def test_invalid_scope_falls_back_to_bot_chat(raw, caplog):
    """A typo must never widen the A2A surface — fail safe, and say so."""
    with caplog.at_level("WARNING"):
        assert _resolve_bot_mode_protocol_scope(raw) == "bot_chat"
    if raw not in ("", None, False, []):
        assert "bot_mode_protocol_scope" in caplog.text


def test_config_default_is_bot_chat():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"]["bot_mode_protocol_scope"] == "bot_chat"


# ── the shared gate ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("title", ["", "Haro task 42", "My research chat", "handoff-12ab"])
def test_gate_default_scope_rejects_non_bot_chat(tmp_path, title):
    agent = _FakeAgent(_managed_home(tmp_path), title=title)
    assert bot_mode_dm.bot_mode_scope_allows(agent, title) is False


def test_gate_default_scope_accepts_bot_chat(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path))
    assert bot_mode_dm.bot_mode_scope_allows(agent, "Bot Chat") is True


@pytest.mark.parametrize("title", ["", "Haro task 42", "Bot Chat", "handoff-12ab"])
def test_gate_all_scope_accepts_any_title(tmp_path, title):
    agent = _FakeAgent(_managed_home(tmp_path), title=title, scope="all")
    assert bot_mode_dm.bot_mode_scope_allows(agent, title) is True


@pytest.mark.parametrize("title", ["Group: room-abc123", "Group: Ceo, CTO, Coding"])
def test_gate_all_scope_still_excludes_group_rooms(tmp_path, title):
    """Group-room plumbing sessions are never an A2A surface."""
    agent = _FakeAgent(_managed_home(tmp_path), title=title, scope="all")
    assert bot_mode_dm.bot_mode_scope_allows(agent, title) is False


def test_gate_all_scope_still_excludes_subagents(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Delegated work", scope="all")
    agent.is_subagent = True
    assert bot_mode_dm.bot_mode_scope_allows(agent, "Delegated work") is False

    depth_agent = _FakeAgent(_managed_home(tmp_path), title="Delegated work", scope="all")
    depth_agent._delegate_depth = 1
    assert bot_mode_dm.bot_mode_scope_allows(depth_agent, "Delegated work") is False


def test_gate_all_scope_still_excludes_cron(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    agent = _FakeAgent(_managed_home(tmp_path), title="Nightly Synthesis", scope="all")
    assert bot_mode_dm.bot_mode_scope_allows(agent, "Nightly Synthesis") is False
    # ...but a cron job that somehow runs IN the canonical Bot Chat keeps the
    # upstream behaviour (the title is the stronger signal).
    assert bot_mode_dm.bot_mode_scope_allows(agent, "Bot Chat") is True


# ── tool injection ───────────────────────────────────────────────────────────


def test_injection_default_scope_unchanged(tmp_path):
    home = _managed_home(tmp_path)
    assert bot_mode_dm.ensure_message_agent_tool(_FakeAgent(home, "Bot Chat")) is True
    ordinary = _FakeAgent(home, "Haro task 42")
    assert bot_mode_dm.ensure_message_agent_tool(ordinary) is False
    assert ordinary.tools == []


def test_injection_all_scope_covers_ordinary_session(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Haro task 42", scope="all")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is True
    assert [t["function"]["name"] for t in agent.tools] == [
        bot_mode_dm.MESSAGE_AGENT_TOOL_NAME
    ]
    assert bot_mode_dm.MESSAGE_AGENT_TOOL_NAME in agent.valid_tool_names


def test_injection_all_scope_still_needs_managed_install(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    agent = _FakeAgent(home, title="Haro task 42", scope="all")
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False
    assert agent.tools == []


def test_injection_all_scope_respects_bot_mode_protocol_toggle(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Haro task 42", scope="all")
    agent._bot_mode_protocol = False
    assert bot_mode_dm.ensure_message_agent_tool(agent) is False


# ── execution-time dispatch ──────────────────────────────────────────────────


def test_dispatch_default_scope_refuses_ordinary_session(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Haro task 42")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result
    assert "Bot Chat" in result["error"]


def test_dispatch_all_scope_passes_the_gate(tmp_path, monkeypatch):
    """Under scope=all the gate no longer refuses; delivery is attempted."""
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, title="Haro task 42", scope="all")

    spawned = {}

    def _fake_spawn(command, label, **kwargs):  # noqa: ARG001
        spawned["command"] = command
        # The real spawner owns the temp DM file; drop it so nothing leaks.
        bot_mode_dm._unlink_dm_file(kwargs.get("dm_file"))
        return json.dumps({"status": "queued", "target": label})

    monkeypatch.setattr(bot_mode_dm, "_spawn_delivery", _fake_spawn, raising=False)
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" not in result, result
    assert spawned["command"], "delivery must actually be spawned"
    assert "-p researcher chat" in spawned["command"]


def test_dispatch_all_scope_still_refuses_unmanaged_install(tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    agent = _FakeAgent(home, title="Haro task 42", scope="all")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result
    assert "not Bot-Mode-managed" in result["error"]


def test_dispatch_all_scope_still_refuses_group_room(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Group: room-abc", scope="all")
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result
    assert "do not retry" in result["error"]


def test_dispatch_all_scope_still_refuses_subagent(tmp_path):
    agent = _FakeAgent(_managed_home(tmp_path), title="Delegated work", scope="all")
    agent.is_subagent = True
    result = json.loads(
        bot_mode_dm.message_agent_tool(target="researcher", message="hi", agent=agent)
    )
    assert "error" in result
    assert "do not retry" in result["error"]


# ── system-prompt section ────────────────────────────────────────────────────


def _section_for(monkeypatch, tmp_path, *, title, scope, managed=True):
    """Build the real prompt parts and report whether the A2A section is in."""
    from run_agent import AIAgent

    home = _managed_home(tmp_path) if managed else (tmp_path / ".hermes")
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))

    class _FakeOpenAI:
        def __init__(self, **kw):
            self.api_key = kw.get("api_key", "test")

        def close(self):
            pass

    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
    agent = AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent._bot_mode_protocol = True
    agent._bot_mode_protocol_scope = scope
    agent._session_title_hint = title

    from agent.system_prompt import build_system_prompt_parts

    blob = " ".join(str(v) for v in build_system_prompt_parts(agent).values())
    return "Messaging other agents" in blob, agent


def test_prompt_section_default_scope(monkeypatch, tmp_path):
    injected, _ = _section_for(monkeypatch, tmp_path, title="Bot Chat", scope="bot_chat")
    assert injected is True


def test_prompt_section_default_scope_skips_ordinary_session(monkeypatch, tmp_path):
    injected, _ = _section_for(
        monkeypatch, tmp_path, title="Haro task 42", scope="bot_chat"
    )
    assert injected is False


def test_prompt_section_all_scope_covers_ordinary_session(monkeypatch, tmp_path):
    injected, agent = _section_for(
        monkeypatch, tmp_path, title="Haro task 42", scope="all"
    )
    assert injected is True
    # Eternal-session support travels with the section, not with the title.
    assert getattr(agent, "_bot_chat_timeless_prompt", False) is True


def test_prompt_section_all_scope_excludes_group_and_unmanaged(monkeypatch, tmp_path):
    injected, _ = _section_for(
        monkeypatch, tmp_path, title="Group: room-abc", scope="all"
    )
    assert injected is False

    injected_unmanaged, _ = _section_for(
        monkeypatch, tmp_path / "plain", title="Haro task 42", scope="all", managed=False
    )
    assert injected_unmanaged is False
