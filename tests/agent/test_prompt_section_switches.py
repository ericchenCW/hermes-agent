"""HERMES_PROMPT_HERMES_HELP / HERMES_PROMPT_STEER_NOTE drop optional built-in prompt sections."""
import pytest

from agent import system_prompt
from agent.system_prompt import build_system_prompt
from tests.agent.test_system_prompt import _make_agent


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return _make_agent(valid_tool_names=["read_file"])


def test_sections_present_by_default(agent, monkeypatch):
    monkeypatch.delenv("HERMES_PROMPT_HERMES_HELP", raising=False)
    monkeypatch.delenv("HERMES_PROMPT_STEER_NOTE", raising=False)
    prompt = build_system_prompt(agent)
    assert "You run on Hermes Agent" in prompt
    assert "Mid-turn user steering" in prompt


def test_sections_absent_when_switched_off(agent, monkeypatch):
    monkeypatch.setenv("HERMES_PROMPT_HERMES_HELP", "0")
    monkeypatch.setenv("HERMES_PROMPT_STEER_NOTE", "0")
    prompt = build_system_prompt(agent)
    assert "You run on Hermes Agent" not in prompt
    assert "Mid-turn user steering" not in prompt


def test_helper_parsing(monkeypatch):
    monkeypatch.setenv("HERMES_PROMPT_STEER_NOTE", "off")
    assert system_prompt._prompt_section_enabled("HERMES_PROMPT_STEER_NOTE") is False
    monkeypatch.delenv("HERMES_PROMPT_STEER_NOTE", raising=False)
    assert system_prompt._prompt_section_enabled("HERMES_PROMPT_STEER_NOTE") is True
