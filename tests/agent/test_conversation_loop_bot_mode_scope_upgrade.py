"""Tests for the "legacy Bot Chat prompt upgrade" branch inside
``agent.conversation_loop._restore_or_build_system_prompt``.

Upstream, a stored system prompt that predates the Bot Mode teammate
protocol (no epoch stamp, no "Messaging other agents" section) gets a
one-time migration rebuild — but only for sessions titled "Bot Chat". That
title check bypassed the shared scope gate (``tools.bot_mode_dm.
bot_mode_scope_allows``) used everywhere else the protocol is gated, so a
stored session that only qualifies under ``bot_mode_protocol_scope: all``
(an ordinary, non-"Bot Chat"-titled session of a Bot-Mode-managed profile)
never got its stale prompt upgraded.

What must hold:
  * scope "all" — a stored, non-"Bot Chat" session missing the protocol
    section triggers the upgrade rebuild;
  * default scope ("bot_chat") — the same session does NOT trigger it;
  * "Bot Chat" title — triggers under both scopes (regression, matches
    upstream behaviour).

Rebase note: upstream dropped the write-only ``agent._bot_capability_refreshed`` flag this test
originally also asserted on; the rebuild is pinned via ``_build_system_prompt`` /
``update_system_prompt`` instead.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent.conversation_loop import _restore_or_build_system_prompt
from tools import bot_mode_probe

# A stored prompt with no epoch stamp and no protocol section — i.e. it
# predates the Bot Mode teammate protocol entirely. No ``Model:``/
# ``Provider:`` lines and no host-info block, so
# ``_stored_prompt_matches_runtime`` trivially considers it fresh.
_LEGACY_STORED_PROMPT = (
    "You are Hermes Agent.\n\n"
    "Conversation started: Tuesday, June 16, 2026\n"
    "Session ID: test-session-id\n"
    "This prompt predates the Bot Mode teammate protocol."
)


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
    def __init__(self, home: Path, title: str, stored_prompt: str):
        self.db_path = str(home / "state.db")
        self._title = title
        self._stored_prompt = stored_prompt
        self.update_system_prompt = MagicMock()

    def get_session(self, _sid):
        return {"system_prompt": self._stored_prompt}

    def get_session_title(self, _sid):
        return self._title


class _FakeAgent:
    """A minimal, non-MagicMock agent stub.

    Deliberately not a ``MagicMock``: several of the gate helpers this
    branch calls through (``_is_subagent_context`` in particular) do
    ``bool(getattr(agent, "is_subagent", False))`` — on an unconfigured
    MagicMock that auto-vivifies a truthy attribute and silently fails the
    gate closed, masking the very branch under test.
    """

    def __init__(self, home: Path, title: str, *, scope: str = "bot_chat"):
        self._session_db = _FakeDB(home, title, _LEGACY_STORED_PROMPT)
        self.session_id = "test-session-id"
        self._session_title_hint = None
        self._bot_mode_protocol = True
        self._bot_mode_protocol_scope = scope
        self.is_subagent = False
        self._delegate_depth = 0
        self.model = ""
        self.provider = ""
        self.platform = "cli"
        self._use_prompt_caching = False
        self._build_system_prompt = MagicMock(return_value="REBUILT_PROMPT_WITH_PROTOCOL")


def test_scope_all_upgrades_stale_ordinary_session(tmp_path):
    """scope=all: a stored, non-"Bot Chat" session missing the protocol
    section is rebuilt."""
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, "Haro task 42", scope="all")

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_called_once_with(None)
    assert agent._cached_system_prompt == "REBUILT_PROMPT_WITH_PROTOCOL"
    agent._session_db.update_system_prompt.assert_called_once_with(
        agent.session_id, "REBUILT_PROMPT_WITH_PROTOCOL"
    )


def test_default_scope_does_not_upgrade_ordinary_session(tmp_path):
    """Default scope (bot_chat): the same stored session is reused
    verbatim — no rebuild."""
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, "Haro task 42", scope="bot_chat")

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_not_called()
    assert agent._cached_system_prompt == _LEGACY_STORED_PROMPT
    agent._session_db.update_system_prompt.assert_not_called()


@pytest.mark.parametrize("scope", ["bot_chat", "all"])
def test_bot_chat_title_upgrades_under_both_scopes(tmp_path, scope):
    """Regression: the canonical "Bot Chat" title still triggers the
    upgrade under both the default and "all" scopes."""
    home = _managed_home(tmp_path)
    agent = _FakeAgent(home, "Bot Chat", scope=scope)

    _restore_or_build_system_prompt(agent, None, [{"role": "user", "content": "hi"}])

    agent._build_system_prompt.assert_called_once_with(None)
    assert agent._cached_system_prompt == "REBUILT_PROMPT_WITH_PROTOCOL"
    agent._session_db.update_system_prompt.assert_called_once_with(
        agent.session_id, "REBUILT_PROMPT_WITH_PROTOCOL"
    )
