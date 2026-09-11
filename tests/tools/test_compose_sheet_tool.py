"""compose_sheet tool: KB-relative path validation and script invocation."""
import json
import os
import subprocess

import pytest

from tools import compose_sheet_tool as cst


@pytest.fixture
def kb(tmp_path, monkeypatch):
    root = tmp_path / "kb"; (root / "guides" / "access").mkdir(parents=True)
    (root / "guides" / "access" / "vpn-user-guide.md").write_text("# vpn\n")
    (tmp_path / "secret.md").write_text("x")
    script = tmp_path / "compose_from_doc.py"; script.write_text("print('图 1: a')\nprint('MEDIA:/opt/data/cache/sheets/x.jpg')\n")
    monkeypatch.setenv("CANWAY_KB_ROOT", str(root)); monkeypatch.setenv("COMPOSE_SHEET_SCRIPT", str(script))
    return root


def test_rejects_escapes_and_non_markdown(kb):
    for bad in ("../secret.md", "/etc/passwd", "guides/access/", "guides/access/nope.md", ""):
        out = json.loads(cst.compose_sheet(bad))
        assert "error" in out, bad


def test_runs_script_with_validated_args(kb, monkeypatch):
    calls = []
    def fake_run(cmd, **kw):
        calls.append(cmd); return subprocess.CompletedProcess(cmd, 0, stdout="图 1: a\nMEDIA:/opt/data/cache/sheets/x.jpg\n", stderr="")
    monkeypatch.setattr(cst.subprocess, "run", fake_run)
    out = cst.compose_sheet("guides/access/vpn-user-guide.md", "windows", "连接", 99)
    assert out.endswith("MEDIA:/opt/data/cache/sheets/x.jpg")
    cmd = calls[0]
    assert cmd[2] == "guides/access/vpn-user-guide.md" and cmd[3:5] == ["--max", str(cst.MAX_STEPS)]
    assert "--os" in cmd and cmd[cmd.index("--os") + 1] == "windows" and "--section" in cmd
    # unknown os value is dropped, not passed through
    calls.clear(); cst.compose_sheet("guides/access/vpn-user-guide.md", "linux")
    assert "--os" not in calls[0]


def test_script_failure_is_reported(kb, monkeypatch):
    monkeypatch.setattr(cst.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom"))
    out = json.loads(cst.compose_sheet("guides/access/vpn-user-guide.md"))
    assert out["error"] == "compose failed" and "boom" in out["detail"]


def test_real_script_end_to_end(kb):
    assert cst.compose_sheet("guides/access/vpn-user-guide.md").endswith("MEDIA:/opt/data/cache/sheets/x.jpg")
    assert cst.check_compose_sheet_requirements() is True


class TestScriptProbing:
    """_script() resolution order: env > haro-kb-search > legacy canway path > glob > legacy default."""

    def test_env_wins_even_if_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cst, "HARO_SCRIPT", str(tmp_path / "haro" / "compose_from_doc.py"))
        monkeypatch.setattr(cst, "DEFAULT_SCRIPT", str(tmp_path / "legacy" / "compose_from_doc.py"))
        monkeypatch.setenv("COMPOSE_SHEET_SCRIPT", str(tmp_path / "env-chosen.py"))
        assert cst._script() == tmp_path / "env-chosen.py"

    def test_new_haro_kb_search_path_hit(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMPOSE_SHEET_SCRIPT", raising=False)
        haro = tmp_path / "skills" / "haro-kb-search" / "scripts" / "compose_from_doc.py"
        haro.parent.mkdir(parents=True); haro.write_text("")
        legacy = tmp_path / "skills" / "canway-it-support-kb" / "scripts" / "compose_from_doc.py"
        monkeypatch.setattr(cst, "HARO_SCRIPT", str(haro))
        monkeypatch.setattr(cst, "DEFAULT_SCRIPT", str(legacy))
        monkeypatch.setattr(cst, "SKILLS_SCRIPT_GLOB", str(tmp_path / "skills" / "*" / "scripts" / "compose_from_doc.py"))
        assert cst._script() == haro

    def test_legacy_canway_path_hit_when_haro_missing(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMPOSE_SHEET_SCRIPT", raising=False)
        haro = tmp_path / "skills" / "haro-kb-search" / "scripts" / "compose_from_doc.py"
        legacy = tmp_path / "skills" / "canway-it-support-kb" / "scripts" / "compose_from_doc.py"
        legacy.parent.mkdir(parents=True); legacy.write_text("")
        monkeypatch.setattr(cst, "HARO_SCRIPT", str(haro))
        monkeypatch.setattr(cst, "DEFAULT_SCRIPT", str(legacy))
        monkeypatch.setattr(cst, "SKILLS_SCRIPT_GLOB", str(tmp_path / "skills" / "*" / "scripts" / "compose_from_doc.py"))
        assert cst._script() == legacy

    def test_glob_hit_when_neither_known_path_exists(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMPOSE_SHEET_SCRIPT", raising=False)
        haro = tmp_path / "skills" / "haro-kb-search" / "scripts" / "compose_from_doc.py"
        legacy = tmp_path / "skills" / "canway-it-support-kb" / "scripts" / "compose_from_doc.py"
        renamed = tmp_path / "skills" / "aaa-renamed-kb" / "scripts" / "compose_from_doc.py"
        renamed.parent.mkdir(parents=True); renamed.write_text("")
        other = tmp_path / "skills" / "zzz-other-kb" / "scripts" / "compose_from_doc.py"
        other.parent.mkdir(parents=True); other.write_text("")
        monkeypatch.setattr(cst, "HARO_SCRIPT", str(haro))
        monkeypatch.setattr(cst, "DEFAULT_SCRIPT", str(legacy))
        monkeypatch.setattr(cst, "SKILLS_SCRIPT_GLOB", str(tmp_path / "skills" / "*" / "scripts" / "compose_from_doc.py"))
        # sorted glob picks the alphabetically-first match
        assert cst._script() == renamed

    def test_nothing_found_falls_back_to_legacy_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("COMPOSE_SHEET_SCRIPT", raising=False)
        haro = tmp_path / "skills" / "haro-kb-search" / "scripts" / "compose_from_doc.py"
        legacy = tmp_path / "skills" / "canway-it-support-kb" / "scripts" / "compose_from_doc.py"
        monkeypatch.setattr(cst, "HARO_SCRIPT", str(haro))
        monkeypatch.setattr(cst, "DEFAULT_SCRIPT", str(legacy))
        monkeypatch.setattr(cst, "SKILLS_SCRIPT_GLOB", str(tmp_path / "skills" / "*" / "scripts" / "compose_from_doc.py"))
        assert cst._script() == legacy
        assert not cst._script().is_file()


class TestKbRootProbing:
    """_kb_root() resolution order: env > default canway path > ALT_KB_ROOT > sole ALT_KB_PARENT/* subdir > legacy default."""

    def _blank(self, tmp_path, monkeypatch):
        """Point every probed location at a nonexistent path under tmp_path."""
        monkeypatch.delenv("CANWAY_KB_ROOT", raising=False)
        monkeypatch.setattr(cst, "DEFAULT_KB_ROOT", str(tmp_path / "opt-kb"))
        monkeypatch.setattr(cst, "ALT_KB_ROOT", str(tmp_path / "knowledge" / "canway-it-support"))
        monkeypatch.setattr(cst, "ALT_KB_PARENT", str(tmp_path / "knowledge"))

    def test_env_wins_even_if_missing(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        monkeypatch.setenv("CANWAY_KB_ROOT", str(tmp_path / "env-kb"))
        assert cst._kb_root() == tmp_path / "env-kb"

    def test_default_canway_path_hit(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        default_root = tmp_path / "opt-kb"; default_root.mkdir()
        monkeypatch.setattr(cst, "DEFAULT_KB_ROOT", str(default_root))
        assert cst._kb_root() == default_root

    def test_alt_kb_root_hit_when_default_missing(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        alt = tmp_path / "knowledge" / "canway-it-support"; alt.mkdir(parents=True)
        monkeypatch.setattr(cst, "ALT_KB_ROOT", str(alt))
        assert cst._kb_root() == alt

    def test_sole_knowledge_subdir_used_when_both_named_paths_absent(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        parent = tmp_path / "knowledge"; parent.mkdir()
        sole = parent / "renamed-kb"; sole.mkdir()
        monkeypatch.setattr(cst, "ALT_KB_PARENT", str(parent))
        assert cst._kb_root() == sole

    def test_multiple_knowledge_subdirs_is_ambiguous_falls_back(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        parent = tmp_path / "knowledge"; parent.mkdir()
        (parent / "kb-a").mkdir(); (parent / "kb-b").mkdir()
        monkeypatch.setattr(cst, "ALT_KB_PARENT", str(parent))
        assert cst._kb_root() == tmp_path / "opt-kb"

    def test_none_found_falls_back_to_legacy_default(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        assert cst._kb_root() == tmp_path / "opt-kb"
        assert not cst._kb_root().is_dir()

    def test_check_requirements_false_when_neither_script_nor_root_exist(self, tmp_path, monkeypatch):
        self._blank(tmp_path, monkeypatch)
        monkeypatch.delenv("COMPOSE_SHEET_SCRIPT", raising=False)
        monkeypatch.setattr(cst, "HARO_SCRIPT", str(tmp_path / "no" / "haro.py"))
        monkeypatch.setattr(cst, "DEFAULT_SCRIPT", str(tmp_path / "no" / "legacy.py"))
        monkeypatch.setattr(cst, "SKILLS_SCRIPT_GLOB", str(tmp_path / "no-skills" / "*" / "scripts" / "compose_from_doc.py"))
        assert cst.check_compose_sheet_requirements() is False
