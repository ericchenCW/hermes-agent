"""Tests for the read allowlist (HERMES_READ_SAFE_ROOTS) and its denylist.

Security fix: a chat-facing Hermes bot running the read-only ``sre-readonly``
toolset (``read_file`` / ``search_files``, sometimes ``terminal``) could be
asked for its own ``/opt/data/config.yaml`` and hand back model endpoints,
runtime token references, IM credential references and internal addresses.

Companion of ``test_file_write_safety.py`` (the write direction).
"""

import json
import os
from pathlib import Path

import pytest

from agent.file_safety import (
    READ_PATH_DENIED_CODE,
    READ_PATH_DENIED_MESSAGE,
    get_command_read_denial,
    get_read_path_denial,
    get_safe_read_roots,
)

DENIAL = {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from "no allowlist, no bypass"."""
    monkeypatch.delenv("HERMES_READ_SAFE_ROOTS", raising=False)
    monkeypatch.delenv("HERMES_READ_SAFE_ROOTS_BYPASS", raising=False)


class TestSafeReadRootsParsing:
    def test_unset_means_no_allowlist(self):
        assert get_safe_read_roots() == set()

    def test_comma_separated(self, monkeypatch, tmp_path: Path):
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", f"{a},{b}")
        assert get_safe_read_roots() == {os.path.realpath(a), os.path.realpath(b)}

    def test_pathsep_separated_and_tilde_expanded(self, monkeypatch, tmp_path: Path):
        a = tmp_path / "a"
        a.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", f"{a}{os.pathsep}~")
        assert os.path.realpath(os.path.expanduser("~")) in get_safe_read_roots()

    def test_blank_entries_dropped(self, monkeypatch, tmp_path: Path):
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", f",,{tmp_path},")
        assert get_safe_read_roots() == {os.path.realpath(tmp_path)}


class TestAllowlist:
    """③ A path inside a configured root reads normally; outside is refused."""

    def test_inside_root_allowed(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        doc = kb / "xxx.md"
        doc.write_text("hello")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_read_path_denial(str(doc)) is None

    def test_outside_root_denied(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        (other / "secret.md").write_text("nope")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_read_path_denial(str(other / "secret.md")) == DENIAL

    def test_dotdot_escape_denied(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("nope")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_read_path_denial(str(kb / ".." / "outside.md")) == DENIAL

    def test_symlink_out_of_root_denied(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        outside = tmp_path / "outside.md"
        outside.write_text("nope")
        link = kb / "link.md"
        link.symlink_to(outside)
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_read_path_denial(str(link)) == DENIAL

    def test_sibling_prefix_is_not_inside_root(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "kb"
        kb.mkdir()
        sibling = tmp_path / "kb-private"
        sibling.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        # "/kb-private" must not pass a naive startswith("/kb") test.
        assert get_read_path_denial(str(sibling / "x.md")) == DENIAL


class TestDenylistWithoutAllowlist:
    """④ With no allowlist configured, the denylist still applies."""

    def test_hermes_config_yaml(self):
        assert get_read_path_denial("/opt/data/config.yaml") == DENIAL

    def test_etc_passwd(self):
        assert get_read_path_denial("/etc/passwd") == DENIAL

    def test_proc(self):
        assert get_read_path_denial("/proc/self/environ") == DENIAL

    def test_hermes_home(self, monkeypatch, tmp_path: Path):
        home = tmp_path / "hermes-home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert get_read_path_denial(str(home / "config.yaml")) == DENIAL

    def test_default_hermes_dir(self):
        assert get_read_path_denial(os.path.expanduser("~/.hermes/config.yaml")) == DENIAL

    @pytest.mark.parametrize("name", [".env", ".env.local", ".envrc", ".env.production"])
    def test_env_files(self, tmp_path: Path, name):
        assert get_read_path_denial(str(tmp_path / name)) == DENIAL

    @pytest.mark.parametrize("name", ["server.key", "server.pem", "CERT.PEM"])
    def test_key_material(self, tmp_path: Path, name):
        assert get_read_path_denial(str(tmp_path / name)) == DENIAL

    def test_ordinary_file_allowed(self, tmp_path: Path):
        assert get_read_path_denial(str(tmp_path / "notes.md")) is None


class TestDenylistStacksOnAllowlist:
    """The denylist keeps applying even to paths inside an allowed root."""

    def test_env_inside_allowed_root_still_denied(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_read_path_denial(str(kb / ".env")) == DENIAL
        assert get_read_path_denial(str(kb / "tls.pem")) == DENIAL


class TestBypass:
    """⑦ HERMES_READ_SAFE_ROOTS_BYPASS=1 opens both layers back up."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_bypass_disables_denylist(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS_BYPASS", value)
        assert get_read_path_denial("/opt/data/config.yaml") is None
        assert get_read_path_denial("/etc/passwd") is None

    def test_bypass_disables_allowlist(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS_BYPASS", "1")
        assert get_read_path_denial(str(tmp_path / "outside.md")) is None

    def test_bypass_disables_command_guard(self, monkeypatch):
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS_BYPASS", "1")
        assert get_command_read_denial("cat /opt/data/config.yaml", "/") is None

    def test_falsey_values_do_not_bypass(self, monkeypatch):
        for value in ("0", "false", "", "no"):
            monkeypatch.setenv("HERMES_READ_SAFE_ROOTS_BYPASS", value)
            assert get_read_path_denial("/etc/passwd") == DENIAL


class TestCommandReadGuard:
    """⑤ The terminal tool cannot walk around the file-tool guard."""

    def test_cat_hermes_config_denied(self):
        assert get_command_read_denial("cat /opt/data/config.yaml", "/tmp") == DENIAL

    def test_cat_allowed_doc(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        (kb / "a.md").write_text("hello")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_command_read_denial(f"cat {kb}/a.md", str(kb)) is None

    @pytest.mark.parametrize("command", [
        "head -5 /opt/data/config.yaml",
        "tail -n 20 /etc/passwd",
        "less /proc/cpuinfo",
        "grep -rn secret /etc",
        "find /etc -name '*.conf'",
        "ls /etc",
        "sed -n '1,5p' /opt/data/config.yaml",
        "awk '{print}' /etc/hosts",
        "python3 -c 'print(open(\"/opt/data/config.yaml\").read())'",
        "cat /tmp/../etc/passwd",
        "cat /opt/data/config.yaml | head -3",
        "echo x && cat /etc/shadow",
    ])
    def test_denied_shapes(self, command):
        assert get_command_read_denial(command, "/tmp") == DENIAL, command

    def test_relative_escape_uses_cwd(self, monkeypatch, tmp_path: Path):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        (tmp_path / "outside.md").write_text("nope")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_command_read_denial("cat ../outside.md", str(kb)) == DENIAL

    def test_unparseable_command_with_denied_path(self):
        # Unbalanced quote: shlex cannot tokenize it, so the conservative
        # tier must still refuse the absolute path it mentions.
        assert get_command_read_denial("cat '/opt/data/config.yaml", "/tmp") == DENIAL

    @pytest.mark.parametrize("command", [
        "echo hello",
        "ls",
        "curl https://example.com/etc/passwd",
        "sed -i 's/foo/bar/' notes.md",
    ])
    def test_allowed_shapes(self, command, tmp_path):
        assert get_command_read_denial(command, str(tmp_path)) is None, command


class TestReadFileToolResponse:
    """① The refusal leaks neither contents nor path existence."""

    def _payload(self, raw):
        return json.loads(raw)

    def test_read_file_denied_payload_shape(self, monkeypatch, tmp_path: Path):
        from tools import file_tools

        existing = tmp_path / "hermes-home"
        existing.mkdir()
        target = existing / "config.yaml"
        target.write_text("bot_secret: supersecretvalue\napi_key: sk-abcdefghijklmnop\n")
        monkeypatch.setenv("HERMES_HOME", str(existing))

        raw = file_tools.read_file_tool(str(target))
        payload = self._payload(raw)
        assert payload == {
            "error": READ_PATH_DENIED_CODE,
            "message": READ_PATH_DENIED_MESSAGE,
        }
        # No content, no path echo, no existence oracle.
        assert "supersecretvalue" not in raw
        assert str(target) not in raw

    def test_missing_and_existing_denied_paths_are_indistinguishable(
        self, monkeypatch, tmp_path: Path
    ):
        from tools import file_tools

        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "there.yaml").write_text("x")
        monkeypatch.setenv("HERMES_HOME", str(home))

        present = file_tools.read_file_tool(str(home / "there.yaml"))
        absent = file_tools.read_file_tool(str(home / "not-there.yaml"))
        assert present == absent

    def test_allowed_root_read_still_works(self, monkeypatch, tmp_path: Path):
        from tools import file_tools

        kb = tmp_path / "knowledge"
        kb.mkdir()
        doc = kb / "xxx.md"
        doc.write_text("plain knowledge base line\n")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))

        payload = self._payload(file_tools.read_file_tool(str(doc)))
        assert "plain knowledge base line" in json.dumps(payload, ensure_ascii=False)


class TestSearchFilesToolResponse:
    """② search_files refuses out-of-root roots the same way."""

    def test_search_outside_root_denied(self, monkeypatch, tmp_path: Path):
        from tools import file_tools

        kb = tmp_path / "knowledge"
        kb.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        (other / "leak.txt").write_text("bot_secret: supersecretvalue")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))

        raw = file_tools.search_tool(pattern="supersecretvalue", path=str(other))
        assert json.loads(raw) == {
            "error": READ_PATH_DENIED_CODE,
            "message": READ_PATH_DENIED_MESSAGE,
        }

    def test_search_dotdot_escape_denied(self, monkeypatch, tmp_path: Path):
        from tools import file_tools

        kb = tmp_path / "knowledge"
        kb.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        raw = file_tools.search_tool(pattern="x", path=str(kb / ".." / "other"))
        assert json.loads(raw)["error"] == READ_PATH_DENIED_CODE

    def test_search_symlink_out_of_root_denied(self, monkeypatch, tmp_path: Path):
        from tools import file_tools

        kb = tmp_path / "knowledge"
        kb.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        link = kb / "escape"
        link.symlink_to(other, target_is_directory=True)
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        raw = file_tools.search_tool(pattern="x", path=str(link))
        assert json.loads(raw)["error"] == READ_PATH_DENIED_CODE


class TestTerminalToolIntegration:
    """⑤ end-to-end through the terminal tool handler itself."""

    def test_terminal_blocks_config_read(self, monkeypatch, tmp_path: Path):
        from tools import terminal_tool

        home = tmp_path / "hermes-home"
        home.mkdir()
        (home / "config.yaml").write_text("bot_secret: supersecretvalue\n")
        monkeypatch.setenv("HERMES_HOME", str(home))

        raw = terminal_tool.terminal_tool(command=f"cat {home}/config.yaml")
        payload = json.loads(raw)
        assert payload["error"] == READ_PATH_DENIED_CODE
        assert payload["message"] == READ_PATH_DENIED_MESSAGE
        assert payload["status"] == "blocked"
        assert "supersecretvalue" not in raw

    def test_terminal_allows_knowledge_read(self, monkeypatch, tmp_path: Path):
        from tools import terminal_tool

        kb = tmp_path / "knowledge"
        kb.mkdir()
        (kb / "a.md").write_text("plain knowledge line\n")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))

        raw = terminal_tool.terminal_tool(command=f"cat {kb}/a.md")
        payload = json.loads(raw)
        assert payload.get("error") != READ_PATH_DENIED_CODE
        assert "plain knowledge line" in payload.get("output", "")
