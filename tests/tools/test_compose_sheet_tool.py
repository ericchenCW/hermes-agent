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
