"""Tests for the "stale identity prompt" rebuild branch inside
``agent.conversation_loop._restore_or_build_system_prompt``.

The system prompt is built once per session and then replayed verbatim on
every later turn for prefix-cache stability.  That is correct for a
running session and wrong for an identity change: after an operator
configures ``agent.identity`` (or after the identity segment's wording
changes) and the container is rebuilt, every pre-existing conversation
keeps answering "I am Hermes Agent, built by Nous Research" from its
archived prompt.  The Bot Chat upgrade branch next to this one only fires
on a missing Bot Chat protocol section, which an identity change never
touches — hence the narrower trigger tested here.

What must hold:
  * identity configured + stored prompt without it → one rebuild, persisted;
  * stored prompt already carrying the current identity → reused verbatim;
  * no identity configured → upstream behaviour, never a rebuild;
  * the rebuild fires at most once per agent (no per-turn cache churn).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from agent.identity_config import AgentIdentity, build_identity_prompt
from tools import bot_mode_probe

_IDENTITY = AgentIdentity(name="IT 小助理", creator="嘉为科技 Haro 平台")

# Pre-identity prompt: no ``Model:``/``Provider:`` lines and no host block,
# so ``_stored_prompt_matches_runtime`` considers it fresh and we reach the
# identity check rather than the runtime-mismatch path.
_LEGACY_STORED_PROMPT = (
    "You are Hermes Agent, built by Nous Research. Be direct.\n\n"
    "Conversation started: Tuesday, June 16, 2026\n"
    "Session ID: test-session-id"
)
_CURRENT_STORED_PROMPT = (
    build_identity_prompt(_IDENTITY) + "\n\nBe direct.\n\n"
    "Conversation started: Tuesday, June 16, 2026\n"
    "Session ID: test-session-id"
)


@pytest.fixture(autouse=True)
def _fresh_probe_cache():
    bot_mode_probe._reset_cache_for_tests()
    yield
    bot_mode_probe._reset_cache_for_tests()


@pytest.fixture(autouse=True)
def _no_ambient_identity(monkeypatch):
    """Env must not leak an identity into the "not configured" cases."""
    for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                "HERMES_IDENTITY_INTRO"):
        monkeypatch.delenv(env, raising=False)


class _FakeDB:
    def __init__(self, home: Path, stored_prompt: str):
        self.db_path = str(home / "state.db")
        self._stored_prompt = stored_prompt
        self.update_system_prompt = MagicMock()

    def get_session(self, _sid):
        return {"system_prompt": self._stored_prompt}

    def get_session_title(self, _sid):
        return "Haro task 42"


class _FakeAgent:
    """Minimal stub — deliberately not a MagicMock (see the Bot Mode scope
    test for why auto-vivifying attributes mask these gates)."""

    def __init__(self, home: Path, stored_prompt: str, identity):
        self._session_db = _FakeDB(home, stored_prompt)
        self.session_id = "test-session-id"
        self._session_title_hint = None
        self._agent_identity = identity
        self._bot_mode_protocol = False
        self._bot_mode_protocol_scope = "bot_chat"
        self.is_subagent = False
        self._delegate_depth = 0
        self.model = ""
        self.provider = ""
        self.platform = "haro"
        self._use_prompt_caching = False
        self._build_system_prompt = MagicMock(return_value="REBUILT_WITH_IDENTITY")


def _agent(tmp_path, stored_prompt, identity):
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    return _FakeAgent(home, stored_prompt, identity)


def test_legacy_session_is_rebuilt_with_the_configured_identity(tmp_path):
    agent = _agent(tmp_path, _LEGACY_STORED_PROMPT, _IDENTITY)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "你是谁"}])

    agent._build_system_prompt.assert_called_once_with(None)
    assert agent._cached_system_prompt == "REBUILT_WITH_IDENTITY"
    agent._session_db.update_system_prompt.assert_called_once_with(
        "test-session-id", "REBUILT_WITH_IDENTITY"
    )


def test_prompt_already_carrying_the_identity_is_reused(tmp_path):
    agent = _agent(tmp_path, _CURRENT_STORED_PROMPT, _IDENTITY)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "你是谁"}])

    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == _CURRENT_STORED_PROMPT
    agent._session_db.update_system_prompt.assert_not_called()


def test_no_identity_configured_never_rebuilds(tmp_path):
    """Stock install: the upstream vendor prompt is reused untouched."""
    agent = _agent(tmp_path, _LEGACY_STORED_PROMPT, None)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == _LEGACY_STORED_PROMPT


def test_refresh_fires_at_most_once_per_agent(tmp_path):
    """Guard against per-turn cache churn if a rebuild can't clear the signal."""
    agent = _agent(tmp_path, _LEGACY_STORED_PROMPT, _IDENTITY)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "你是谁"}])
    assert agent._identity_prompt_refreshed is True

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "再问一次"}])
    agent._build_system_prompt.assert_called_once()


def test_db_write_failure_does_not_break_the_turn(tmp_path):
    agent = _agent(tmp_path, _LEGACY_STORED_PROMPT, _IDENTITY)
    agent._session_db.update_system_prompt.side_effect = RuntimeError("disk full")

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "你是谁"}])

    assert agent._cached_system_prompt == "REBUILT_WITH_IDENTITY"
