"""Tests for the white-label ``agent.identity`` config (agent/identity_config.py).

Default (all fields empty) must keep the upstream Hermes/Nous identity
segment byte-for-byte; a configured name must REPLACE it.
"""

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.identity_config import (
    AgentIdentity,
    build_identity_prompt,
    resolve_identity,
    scrub_vendor_names,
    strip_builtin_identity,
)
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from agent.system_prompt import build_system_prompt_parts
from hermes_cli.config_defaults import DEFAULT_CONFIG
from hermes_cli.default_soul import DEFAULT_SOUL_MD

# The identity segment is what we assert on. Other prompt text may legitimately
# contain the substring "hermes" inside identifiers/paths (hermes_cli,
# HERMES_HOME, hermes-agent.nousresearch.com), so the vendor check uses the
# same word/path boundaries as the scrubber instead of a bare `in`.
_VENDOR_PROSE = re.compile(r"(?<![\w./\-])(Hermes|Nous)(?![\w./\-])", re.IGNORECASE)


def _without_constraint(text):
    """Drop the identity block's hard-constraint sentence, which deliberately
    keeps the literal vendor names ("never call yourself Hermes")."""
    return re.sub(r"身份约束（[^\n]*", "", text)


def _make_agent(**overrides):
    base = dict(
        load_soul_identity=False,
        skip_context_files=False,
        valid_tool_names=[],
        _task_completion_guidance=False,
        _tool_use_enforcement=False,
        _environment_probe=False,
        _kanban_worker_guidance="",
        _memory_store=None,
        _memory_manager=None,
        model="",
        provider="",
        platform="",
        pass_session_id=False,
        session_id="",
        _emit_status=lambda *_a, **_k: None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _stable(agent, soul=""):
    with (
        patch("agent.prompt_builder.load_soul_md", return_value=soul),
        patch("agent.prompt_builder.build_environment_hints", return_value=""),
        patch("agent.prompt_builder.build_context_files_prompt", return_value=""),
    ):
        return build_system_prompt_parts(agent)["stable"]


class TestResolveIdentity:
    def test_defaults_are_empty_and_off(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        section = DEFAULT_CONFIG["agent"]["identity"]
        assert section == {"name": "", "creator": "", "intro": ""}
        assert resolve_identity(section) is None
        assert resolve_identity(None) is None

    def test_config_values(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        ident = resolve_identity(
            {"name": "IT 小助理", "creator": "嘉为科技 Haro 平台", "intro": "我帮你解决 IT 问题。"}
        )
        assert ident == AgentIdentity(
            name="IT 小助理", creator="嘉为科技 Haro 平台", intro="我帮你解决 IT 问题。"
        )

    def test_env_overrides_config_per_field(self, monkeypatch):
        monkeypatch.setenv("HERMES_IDENTITY_NAME", "运维助手")
        monkeypatch.delenv("HERMES_IDENTITY_CREATOR", raising=False)
        monkeypatch.setenv("HERMES_IDENTITY_INTRO", "env 自我介绍")
        ident = resolve_identity(
            {"name": "IT 小助理", "creator": "嘉为科技 Haro 平台", "intro": "config 自我介绍"}
        )
        assert ident.name == "运维助手"          # env wins
        assert ident.creator == "嘉为科技 Haro 平台"  # falls back to config
        assert ident.intro == "env 自我介绍"

    def test_env_alone_enables_without_config(self, monkeypatch):
        monkeypatch.setenv("HERMES_IDENTITY_NAME", "IT 小助理")
        monkeypatch.setenv("HERMES_IDENTITY_CREATOR", "嘉为科技 Haro 平台")
        monkeypatch.delenv("HERMES_IDENTITY_INTRO", raising=False)
        ident = resolve_identity(None)
        assert ident is not None and ident.name == "IT 小助理"

    @pytest.mark.parametrize("bad", [{"name": []}, {"name": {}}, {"name": "   "}, "junk"])
    def test_malformed_stays_off(self, monkeypatch, bad):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        assert resolve_identity(bad) is None


class TestIdentitySegment:
    def test_contains_answer_line_and_hard_constraint(self):
        seg = build_identity_prompt(
            AgentIdentity(name="IT 小助理", creator="嘉为科技 Haro 平台")
        )
        assert "我是 IT 小助理，由 嘉为科技 Haro 平台 提供" in seg
        assert "不要自称 Hermes 或 Nous Research" in seg
        assert "模型供应商" in seg

    def test_intro_is_used_verbatim(self):
        seg = build_identity_prompt(
            AgentIdentity(name="IT 小助理", creator="嘉为科技", intro="我是公司 IT 台的值班助手。")
        )
        assert seg.startswith("我是公司 IT 台的值班助手。")

    def test_generated_intro_without_config_intro(self):
        seg = build_identity_prompt(AgentIdentity(name="IT 小助理", creator="嘉为科技"))
        assert seg.startswith("你是 IT 小助理，由 嘉为科技 提供的智能助手。")


class TestStripAndScrub:
    def test_strip_drops_vendor_sentence_keeps_behavior(self):
        stripped = strip_builtin_identity(DEFAULT_SOUL_MD)
        assert "Hermes" not in stripped
        assert "Nous Research" not in stripped
        assert "Be direct" in stripped
        assert "Depth is earned" in stripped

    def test_scrub_leaves_identifiers_and_urls_alone(self):
        ident = AgentIdentity(name="IT 小助理", creator="嘉为科技")
        text = (
            "You run on Hermes Agent (by Nous Research). See "
            "https://hermes-agent.nousresearch.com/docs, hermes_cli/config.py "
            "and $HERMES_HOME."
        )
        out = scrub_vendor_names(text, ident)
        assert out.startswith("You run on IT 小助理 (by 嘉为科技).")
        assert "https://hermes-agent.nousresearch.com/docs" in out
        assert "hermes_cli/config.py" in out
        assert "$HERMES_HOME" in out


class TestSystemPromptWiring:
    def test_default_keeps_upstream_identity(self, monkeypatch):
        """Regression: no identity configured → upstream segment verbatim."""
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        stable = _stable(_make_agent(_agent_identity=None))
        assert DEFAULT_AGENT_IDENTITY in stable
        assert "You run on Hermes Agent (by Nous Research)" in stable

    def test_agent_without_attribute_keeps_upstream_identity(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        assert DEFAULT_AGENT_IDENTITY in _stable(_make_agent())

    def test_configured_identity_replaces_builtin(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        ident = resolve_identity(
            {"name": "IT 小助理", "creator": "嘉为科技 Haro 平台", "intro": ""}
        )
        stable = _stable(_make_agent(_agent_identity=ident), soul=DEFAULT_SOUL_MD)

        # Replacement, not append: the ONLY vendor prose left is inside the
        # hard-constraint sentence, which must keep the literal names.
        assert DEFAULT_AGENT_IDENTITY not in stable
        assert "我是 IT 小助理，由 嘉为科技 Haro 平台 提供" in stable
        assert "不要自称 Hermes 或 Nous Research" in stable
        assert not _VENDOR_PROSE.findall(_without_constraint(stable))
        # SOUL.md's behavior half survives the identity swap.
        assert "Depth is earned" in stable

    def test_identity_replaces_default_when_no_soul(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        ident = resolve_identity({"name": "IT 小助理", "creator": "嘉为科技 Haro 平台"})
        stable = _stable(_make_agent(_agent_identity=ident), soul="")
        assert not _VENDOR_PROSE.findall(_without_constraint(stable))
        assert "我是 IT 小助理，由 嘉为科技 Haro 平台 提供" in stable

    def test_intro_reaches_the_prompt(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        ident = resolve_identity(
            {"name": "IT 小助理", "creator": "嘉为科技", "intro": "我是公司 IT 台的值班助手。"}
        )
        stable = _stable(_make_agent(_agent_identity=ident))
        assert "我是公司 IT 台的值班助手。" in stable

    def test_env_only_identity_applies_without_agent_init(self, monkeypatch):
        """Container injection: env alone white-labels an agent stub."""
        monkeypatch.setenv("HERMES_IDENTITY_NAME", "IT 小助理")
        monkeypatch.setenv("HERMES_IDENTITY_CREATOR", "嘉为科技 Haro 平台")
        monkeypatch.delenv("HERMES_IDENTITY_INTRO", raising=False)
        stable = _stable(_make_agent(), soul=DEFAULT_SOUL_MD)
        assert "我是 IT 小助理，由 嘉为科技 Haro 平台 提供" in stable
        assert not _VENDOR_PROSE.findall(_without_constraint(stable))

    def test_user_authored_soul_persona_is_preserved(self, monkeypatch):
        for env in ("HERMES_IDENTITY_NAME", "HERMES_IDENTITY_CREATOR",
                    "HERMES_IDENTITY_INTRO"):
            monkeypatch.delenv(env, raising=False)
        ident = resolve_identity({"name": "IT 小助理", "creator": "嘉为科技"})
        soul = "回答要求：先给结论，再给步骤。不要输出内部推理。"
        stable = _stable(_make_agent(_agent_identity=ident), soul=soul)
        assert soul in stable
