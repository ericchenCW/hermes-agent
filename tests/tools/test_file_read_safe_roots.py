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
    """④ With no allowlist configured, only the tier ① denials apply.

    Round 3 split the denylist in two. Tier ① (credential & privacy) is
    unconditional; tier ② (whole directory trees such as ``/etc``) is skipped
    entirely when ``HERMES_READ_SAFE_ROOTS`` is unset, so a CLI or standalone
    gateway fork keeps upstream read behaviour. See
    ``TestTwoTierMatrix`` for the full two-tier behaviour matrix.
    """

    def test_hermes_config_yaml(self):
        assert get_read_path_denial("/opt/data/config.yaml") == DENIAL

    def test_etc_passwd_allowed_without_allowlist(self):
        # Tier ②: gated on HERMES_READ_SAFE_ROOTS being set.
        assert get_read_path_denial("/etc/passwd") is None

    def test_proc(self):
        assert get_read_path_denial("/proc/self/environ") == DENIAL

    @pytest.mark.parametrize("pid", ["1", "42", "99999"])
    def test_proc_pid_environ(self, pid):
        assert get_read_path_denial(f"/proc/{pid}/environ") == DENIAL

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

    def test_config_yaml_outside_hermes_home_allowed(self, monkeypatch, tmp_path: Path):
        # Round 3: the basename is no longer denied globally — a knowledge-base
        # article that happens to be called config.yaml is ordinary content.
        home = tmp_path / "hermes-home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        assert get_read_path_denial(str(tmp_path / "kb" / "config.yaml")) is None


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
            assert get_read_path_denial("/opt/data/config.yaml") == DENIAL


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
        "sed -n '1,5p' /opt/data/config.yaml",
        "python3 -c 'print(open(\"/opt/data/config.yaml\").read())'",
        "cat /opt/data/config.yaml | head -3",
        "cat /proc/self/environ",
        "echo x && cat /proc/1/environ",
        "cat /srv/app/.env",
        "head -1 /srv/certs/server.key",
    ])
    def test_denied_shapes(self, command):
        """Tier ① shapes: refused whether or not an allowlist is configured."""
        assert get_command_read_denial(command, "/tmp") == DENIAL, command

    _TIER2_COMMANDS = [
        "tail -n 20 /etc/passwd",
        "less /proc/cpuinfo",
        "grep -rn secret /etc",
        "find /etc -name '*.conf'",
        "ls /etc",
        "awk '{print}' /etc/hosts",
        "cat /tmp/../etc/passwd",
        "echo x && cat /etc/shadow",
    ]

    @pytest.mark.parametrize("command", _TIER2_COMMANDS)
    def test_directory_tier_shapes_allowed_without_allowlist(self, command, tmp_path):
        """Tier ② shapes read normally when no allowlist is configured."""
        assert get_command_read_denial(command, str(tmp_path)) is None, command

    @pytest.mark.parametrize("command", _TIER2_COMMANDS)
    def test_directory_tier_shapes_denied_with_allowlist(
        self, command, monkeypatch, tmp_path
    ):
        kb = tmp_path / "knowledge"
        kb.mkdir()
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS", str(kb))
        assert get_command_read_denial(command, str(kb)) == DENIAL, command

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
        (home / ".env").write_text("x")
        monkeypatch.setenv("HERMES_HOME", str(home))

        present = file_tools.read_file_tool(str(home / ".env"))
        absent = file_tools.read_file_tool(str(home / ".env.not-there"))
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


class TestTwoTierMatrix:
    """⑥ Round 3: tier ① is unconditional, tier ② is allowlist-gated.

    This tree is forked for plain CLI and standalone-gateway deployments that
    never set ``HERMES_READ_SAFE_ROOTS``. There, blanket-denying ``/etc`` broke
    ordinary operator work for no security gain — but the credential and
    privacy files must stay unreadable regardless. The two blocks below are the
    agreed behaviour matrix.
    """

    @pytest.fixture
    def home(self, monkeypatch, tmp_path: Path) -> Path:
        """A Haro-shaped ``HERMES_HOME`` with NO allowlist configured."""
        home = tmp_path / "opt-data"
        for sub in ("logs", "memories", "sessions", "skills/x"):
            (home / sub).mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    # --- no allowlist: tier ① only -----------------------------------------

    def test_etc_hosts_allowed(self, home):
        assert get_read_path_denial("/etc/hosts") is None

    def test_hermes_skill_allowed(self, home):
        assert get_read_path_denial(
            os.path.expanduser("~/.hermes/skills/x/SKILL.md")
        ) is None

    def test_hermes_home_log_allowed(self, home):
        assert get_read_path_denial(str(home / "logs" / "x.log")) is None

    def test_hermes_home_skill_allowed(self, home):
        assert get_read_path_denial(str(home / "skills" / "x" / "SKILL.md")) is None

    def test_hermes_home_config_denied(self, home):
        assert get_read_path_denial(str(home / "config.yaml")) == DENIAL

    @pytest.mark.parametrize(
        "name", ["config.yaml", "state.db", "auth.json", "channel_directory.json", ".env"]
    )
    def test_hermes_home_credential_files_denied(self, home, name):
        assert get_read_path_denial(str(home / name)) == DENIAL

    def test_default_hermes_dotenv_denied(self, home):
        assert get_read_path_denial(os.path.expanduser("~/.hermes/.env")) == DENIAL

    def test_proc_environ_denied(self, home):
        assert get_read_path_denial("/proc/self/environ") == DENIAL
        assert get_read_path_denial("/proc/1234/environ") == DENIAL

    def test_proc_environ_normpath_spelling_denied(self, home):
        # The guard checks realpath AND normpath, so a ``..`` spelling of the
        # same file cannot dodge the regex.
        assert get_read_path_denial("/proc/self/../self/environ") == DENIAL

    def test_hermes_memories_denied(self, home):
        assert get_read_path_denial(str(home / "memories" / "a.md")) == DENIAL

    def test_hermes_sessions_denied(self, home):
        assert get_read_path_denial(str(home / "sessions" / "s.json")) == DENIAL

    def test_other_proc_entries_allowed(self, home):
        assert get_read_path_denial("/proc/cpuinfo") is None

    # --- allowlist configured: round-2 behaviour, unchanged -----------------

    @pytest.fixture
    def allowlisted(self, monkeypatch, tmp_path: Path) -> dict:
        """``/knowledge,/opt/data/skills,/out`` against a Haro-shaped home."""
        home = tmp_path / "opt-data"
        (home / "skills" / "x").mkdir(parents=True)
        (home / "logs").mkdir()
        kb = tmp_path / "knowledge"
        (kb / "x").mkdir(parents=True)
        out = tmp_path / "out"
        out.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv(
            "HERMES_READ_SAFE_ROOTS", f"{kb},{home / 'skills'},{out}"
        )
        return {"home": home, "kb": kb, "out": out}

    def test_allowlisted_skill_allowed(self, allowlisted):
        target = allowlisted["home"] / "skills" / "x" / "SKILL.md"
        assert get_read_path_denial(str(target)) is None

    def test_env_in_allowlisted_skill_denied(self, allowlisted):
        target = allowlisted["home"] / "skills" / "x" / ".env"
        assert get_read_path_denial(str(target)) == DENIAL

    def test_hermes_home_logs_denied_with_allowlist(self, allowlisted):
        target = allowlisted["home"] / "logs" / "x.log"
        assert get_read_path_denial(str(target)) == DENIAL

    @pytest.mark.parametrize("path", ["/etc/hosts", "/etc/passwd", "/proc/cpuinfo"])
    def test_system_trees_denied_with_allowlist(self, allowlisted, path):
        assert get_read_path_denial(path) == DENIAL

    def test_kb_document_named_config_yaml_allowed(self, allowlisted):
        # The whole point of round 3's narrowing: a knowledge-base article that
        # happens to be called config.yaml is ordinary content.
        target = allowlisted["kb"] / "x" / "config.yaml"
        assert get_read_path_denial(str(target)) is None

    # --- terminal command guard follows the same two tiers ------------------

    def test_terminal_cat_etc_hosts_allowed_without_allowlist(self, home):
        assert get_command_read_denial("cat /etc/hosts", str(home)) is None

    def test_terminal_cat_hermes_config_denied_without_allowlist(self, home):
        assert get_command_read_denial(
            f"cat {home}/config.yaml", str(home)
        ) == DENIAL

    def test_terminal_cat_proc_environ_denied_without_allowlist(self, home):
        assert get_command_read_denial("cat /proc/self/environ", str(home)) == DENIAL

    def test_terminal_cat_hermes_memories_denied_without_allowlist(self, home):
        assert get_command_read_denial(
            f"cat {home}/memories/a.md", str(home)
        ) == DENIAL

    def test_terminal_cat_hermes_log_allowed_without_allowlist(self, home):
        assert get_command_read_denial(f"cat {home}/logs/x.log", str(home)) is None

    def test_terminal_cat_etc_hosts_denied_with_allowlist(self, allowlisted):
        assert get_command_read_denial(
            "cat /etc/hosts", str(allowlisted["kb"])
        ) == DENIAL
