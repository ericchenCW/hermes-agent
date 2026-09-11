"""compose_sheet tool: KB-relative path validation and script invocation."""
import io
import json
import os
import random
import subprocess

import pytest
from PIL import Image

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


def _make_big_jpeg(path, size=(3000, 6000)):
    """Write a large, hard-to-compress JPEG (noisy speckles beat flat-color compression)."""
    im = Image.new("RGB", size, color=(120, 60, 200))
    px = im.load()
    rng = random.Random(1)
    for x in range(0, size[0], 7):
        for y in range(0, size[1], 7):
            px[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    im.save(path, format="JPEG", quality=95)
    return path.stat().st_size


class TestMediaPostprocessing:
    """Oversized MEDIA:<path>.jpg output is recompressed under COMPOSE_SHEET_MAX_BYTES."""

    def test_oversized_image_is_compressed_to_dash_c_file_under_limit(self, tmp_path, monkeypatch):
        sheets = tmp_path / "sheets"; sheets.mkdir()
        img = sheets / "steps-1.jpg"
        orig_size = _make_big_jpeg(img)
        assert orig_size > cst.COMPOSE_SHEET_MAX_BYTES
        out = cst._postprocess_media(f"图 1: a\nMEDIA:{img}")
        lines = out.splitlines()
        assert lines[-1] == f"MEDIA:{img.with_name('steps-1-c.jpg')}"
        assert any(line.startswith("已压缩：原") for line in lines)
        compressed = img.with_name("steps-1-c.jpg")
        assert compressed.is_file()
        assert compressed.stat().st_size <= cst.COMPOSE_SHEET_MAX_BYTES
        # original is left untouched
        assert img.is_file() and img.stat().st_size == orig_size

    def test_multiple_media_lines_each_handled_independently(self, tmp_path, monkeypatch):
        sheets = tmp_path / "sheets"; sheets.mkdir()
        big = sheets / "steps-big.jpg"
        _make_big_jpeg(big)
        small = sheets / "steps-small.jpg"
        Image.new("RGB", (200, 200), color=(1, 2, 3)).save(small, format="JPEG", quality=90)
        assert small.stat().st_size <= cst.COMPOSE_SHEET_MAX_BYTES
        out = cst._postprocess_media(f"图 1: a\nMEDIA:{big}\n图 2: b\nMEDIA:{small}")
        media_lines = [l for l in out.splitlines() if l.startswith("MEDIA:")]
        assert media_lines == [f"MEDIA:{big.with_name('steps-big-c.jpg')}", f"MEDIA:{small}"]

    def test_within_limit_passes_through_unchanged(self, tmp_path):
        sheets = tmp_path / "sheets"; sheets.mkdir()
        img = sheets / "steps-2.jpg"
        Image.new("RGB", (400, 400), color=(9, 9, 9)).save(img, format="JPEG", quality=90)
        original = f"图 1: a\nMEDIA:{img}"
        assert cst._postprocess_media(original) == original

    def test_missing_media_file_passes_through_unchanged(self):
        original = "图 1: a\nMEDIA:/opt/data/cache/sheets/does-not-exist.jpg"
        assert cst._postprocess_media(original) == original

    def test_no_images_output_is_not_touched(self):
        assert cst._postprocess_media("NO_IMAGES") == "NO_IMAGES"

    def test_still_too_big_after_compression_adds_warning_but_keeps_smallest_result(self, tmp_path, monkeypatch):
        sheets = tmp_path / "sheets"; sheets.mkdir()
        img = sheets / "steps-huge.jpg"
        _make_big_jpeg(img)
        monkeypatch.setattr(cst, "MIN_LONG_EDGE", 5900)  # force an unreachable floor -> stays over budget
        out = cst._postprocess_media(f"MEDIA:{img}", max_bytes=1)
        lines = out.splitlines()
        assert any(line.startswith("WARNING:") for line in lines)
        assert lines[-1] == f"MEDIA:{img.with_name('steps-huge-c.jpg')}"
        assert img.with_name("steps-huge-c.jpg").is_file()

    def test_compose_sheet_end_to_end_runs_postprocessing(self, kb, tmp_path, monkeypatch):
        sheets = tmp_path / "sheets"; sheets.mkdir()
        img = sheets / "steps-e2e.jpg"
        _make_big_jpeg(img)

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout=f"图 1: a\nMEDIA:{img}\n", stderr="")

        monkeypatch.setattr(cst.subprocess, "run", fake_run)
        out = cst.compose_sheet("guides/access/vpn-user-guide.md")
        assert out.splitlines()[-1] == f"MEDIA:{img.with_name('steps-e2e-c.jpg')}"
