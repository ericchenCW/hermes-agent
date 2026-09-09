"""Round-2 read guard: allowlist precedence, audit trail, export & read quota.

Round 1 (``test_file_read_safe_roots.py``) put ``HERMES_READ_SAFE_ROOTS`` in
front of ``read_file`` / ``search_files`` / ``terminal``.  Deploying it against
the real Haro container surfaced three gaps, each covered here:

1. ``HERMES_HOME=/opt/data`` made the *whole* ``/opt/data`` tree denied, which
   swallowed ``/opt/data/skills`` — an explicitly allowlisted root.  Explicit
   allowlist roots now outrank directory-level denies, while file-level denies
   (``.env``, ``*.key``, ``config.yaml``, ``state.db`` …) still win everywhere.
2. Refusals left no trace.  Every refusal now appends one JSON line to
   ``$HERMES_HOME/logs/readguard.jsonl`` — metadata only, never content.
3. Reading files one at a time was bounded by nothing, and nothing stopped
   ``tar czf /out/kb.tgz /knowledge``.  Hence the bulk-export guard and the
   per-turn knowledge read quota.
"""

import json
import os
from pathlib import Path

import pytest

from agent.file_safety import (
    KB_EXPORT_DENIED_CODE,
    KB_READ_QUOTA_CODE,
    READ_PATH_DENIED_CODE,
    READ_PATH_DENIED_MESSAGE,
    READGUARD_REASON_EXPORT,
    READGUARD_REASON_FILE,
    READGUARD_REASON_PATH,
    READGUARD_REASON_QUOTA,
    check_kb_read_quota,
    classify_read_path_denial,
    get_command_export_denial,
    get_kb_roots,
    get_read_path_denial,
    get_readguard_log_path,
    reset_all_kb_read_quotas,
    reset_kb_read_quota,
)

DENIAL = {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "HERMES_READ_SAFE_ROOTS",
        "HERMES_READ_SAFE_ROOTS_BYPASS",
        "HERMES_KB_ROOTS",
        "HERMES_KB_READ_PER_TURN",
        "HERMES_SESSION_ID",
        "HERMES_SESSION_USER_ID",
        "HERMES_SESSION_PLATFORM",
    ):
        monkeypatch.delenv(name, raising=False)
    # Session ContextVars leak between tests in one process and mask the
    # os.environ fallback; start every test from "no session bound".
    from gateway.session_context import clear_session_vars

    clear_session_vars([])
    reset_all_kb_read_quotas()
    yield
    reset_all_kb_read_quotas()


@pytest.fixture
def deployment(monkeypatch, tmp_path: Path):
    """Reproduce the Haro container layout.

    ``HERMES_HOME`` is the whole ``opt-data`` tree, and ``skills`` inside it is
    handed to the bot as a read root alongside ``/knowledge``, ``/out`` and
    ``/drafts``.
    """
    home = tmp_path / "opt-data"
    (home / "skills" / "x").mkdir(parents=True)
    (home / "logs").mkdir()
    (home / "memories").mkdir()
    (home / "sessions").mkdir()
    (home / "skills" / "x" / "SKILL.md").write_text("skill body\n")
    (home / "skills" / "x" / ".env").write_text("TOKEN=supersecretvalue\n")
    (home / "config.yaml").write_text("bot_secret: supersecretvalue\n")
    (home / "state.db").write_text("sqlite-ish\n")
    (home / "auth.json").write_text('{"token": "supersecretvalue"}\n')
    (home / "channel_directory.json").write_text('{"chat": "supersecretvalue"}\n')
    (home / ".env").write_text("API_KEY=supersecretvalue\n")

    kb = tmp_path / "knowledge"
    (kb / "canway-it-support").mkdir(parents=True)
    (kb / "canway-it-support" / "a.md").write_text("kb line\n")
    out = tmp_path / "out"
    out.mkdir()
    (out / "report").mkdir()

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(
        "HERMES_READ_SAFE_ROOTS", f"{kb},{home / 'skills'},{out}"
    )
    return {"home": home, "kb": kb, "out": out}


def _log_lines(home: Path) -> list[dict]:
    log = home / "logs" / "readguard.jsonl"
    if not log.exists():
        return []
    return [
        json.loads(line)
        for line in log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# 1. Allowlist precedence
# ---------------------------------------------------------------------------


class TestAllowlistBeatsDirectoryDeny:
    def test_allowlisted_subtree_of_hermes_home_is_readable(self, deployment):
        target = deployment["home"] / "skills" / "x" / "SKILL.md"
        assert classify_read_path_denial(str(target)) is None
        assert get_read_path_denial(str(target)) is None

    @pytest.mark.parametrize(
        "name",
        ["config.yaml", "state.db", "auth.json", "channel_directory.json", ".env"],
    )
    def test_named_hermes_home_files_denied(self, deployment, name):
        assert get_read_path_denial(str(deployment["home"] / name)) == DENIAL

    @pytest.mark.parametrize("sub", ["memories", "sessions", "logs"])
    def test_named_hermes_home_dirs_denied(self, deployment, sub):
        target = deployment["home"] / sub / "anything.txt"
        assert get_read_path_denial(str(target)) == DENIAL

    def test_readguard_log_itself_denied(self, deployment):
        assert get_read_path_denial(get_readguard_log_path()) == DENIAL

    def test_file_level_deny_still_wins_inside_allowlisted_root(self, deployment):
        # .env inside the *allowlisted* skills root stays refused.
        target = deployment["home"] / "skills" / "x" / ".env"
        assert classify_read_path_denial(str(target)) == READGUARD_REASON_FILE
        assert get_read_path_denial(str(target)) == DENIAL

    def test_env_inside_knowledge_root_denied(self, deployment):
        assert (
            classify_read_path_denial(str(deployment["kb"] / ".env"))
            == READGUARD_REASON_FILE
        )

    @pytest.mark.parametrize(
        "path",
        [
            "/proc/self/environ",
            "/proc/cpuinfo",
            "/sys/class/net",
            "/sys/kernel/debug/x",
            "/dev/mem",
            "/dev/sda1",
            "/etc/passwd",
        ],
    )
    def test_system_trees_denied(self, deployment, path):
        assert get_read_path_denial(path) == DENIAL

    def test_reason_codes_distinguish_the_two_tiers(self, deployment):
        # Tier ① — credential & privacy, always on.
        assert (
            classify_read_path_denial(str(deployment["home"] / "config.yaml"))
            == READGUARD_REASON_FILE
        )
        # Round 3 promoted memories/ and sessions/ from the directory tier to
        # the always-on tier, so they now report ``denied_file``.
        assert (
            classify_read_path_denial(str(deployment["home"] / "memories" / "m.md"))
            == READGUARD_REASON_FILE
        )
        assert (
            classify_read_path_denial(str(deployment["home"] / "sessions" / "s.json"))
            == READGUARD_REASON_FILE
        )
        # Tier ② — directory-level, gated on the allowlist being configured.
        assert (
            classify_read_path_denial(str(deployment["home"] / "logs" / "x.log"))
            == READGUARD_REASON_PATH
        )
        assert classify_read_path_denial("/etc/passwd") == READGUARD_REASON_PATH

    def test_model_payload_stays_uniform_across_reasons(self, deployment):
        a = get_read_path_denial(str(deployment["home"] / "config.yaml"))
        b = get_read_path_denial(str(deployment["home"] / "logs" / "x.log"))
        c = get_read_path_denial("/proc/self/environ")
        assert a == b == c == DENIAL


class TestSearchFilesDoesNotLeakExistence:
    def test_denied_dir_list_search_is_indistinguishable(self, deployment, monkeypatch):
        from tools import file_tools

        present = file_tools.search_tool(
            pattern="*", target="filename", path=str(deployment["home"] / "sessions")
        )
        absent = file_tools.search_tool(
            pattern="*", target="filename", path=str(deployment["home"] / "no-such-dir")
        )
        assert json.loads(present) == {
            "error": READ_PATH_DENIED_CODE,
            "message": READ_PATH_DENIED_MESSAGE,
        }
        assert present == absent


# ---------------------------------------------------------------------------
# 2. readguard.jsonl audit trail
# ---------------------------------------------------------------------------


_REQUIRED_FIELDS = {"ts", "tool", "path", "session", "subject", "platform", "reason"}


class TestReadguardAudit:
    def test_read_file_denial_writes_one_full_line(self, deployment):
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import file_tools

        # Bind the identity the way the gateway does — the ContextVars mask the
        # os.environ fallback, so setenv alone is not a faithful stand-in.
        tokens = set_session_vars(
            platform="wecom", session_id="sess-1", user_id="wo-user-1"
        )
        try:
            target = deployment["home"] / "config.yaml"
            raw = file_tools.read_file_tool(str(target))
            assert json.loads(raw)["error"] == READ_PATH_DENIED_CODE
        finally:
            clear_session_vars(tokens)

        lines = _log_lines(deployment["home"])
        assert len(lines) == 1
        row = lines[0]
        assert set(row) == _REQUIRED_FIELDS
        assert row["tool"] == "read_file"
        assert row["reason"] == READGUARD_REASON_FILE
        assert row["path"] == os.path.normpath(str(target))
        assert row["session"] == "sess-1"
        assert row["subject"] == "wo-user-1"
        assert row["platform"] == "wecom"
        assert row["ts"].endswith("Z")
        # metadata only — never the file's content
        assert "supersecretvalue" not in json.dumps(row)

    def test_missing_identity_recorded_as_unknown(self, deployment):
        from tools import file_tools

        file_tools.read_file_tool(str(deployment["home"] / "state.db"))
        row = _log_lines(deployment["home"])[0]
        assert row["session"] == "unknown"
        assert row["subject"] == "unknown"
        assert row["platform"] == "unknown"

    def test_gateway_session_shape_identity_not_unknown(self, deployment):
        """Mirror ``HermesGateway._set_session_env`` (gateway/run.py), which
        binds the real inbound-message identity fields for every platform
        adapter (WeCom included) — it passes ``session_key=`` (never
        ``session_id=``) and ``user_id=`` from ``MessageSource.user_id``.

        ``get_readguard_identity()`` (agent/file_safety.py) reads
        ``HERMES_SESSION_ID`` first and falls back to ``HERMES_SESSION_KEY``
        for the audit "session" field, so a real gateway turn — which never
        sets ``HERMES_SESSION_ID`` — must still resolve to a non-"unknown"
        session via that fallback.
        """
        from gateway.session_context import clear_session_vars, set_session_vars
        from tools import file_tools

        tokens = set_session_vars(
            platform="wecom",
            session_key="wecom:corpid-1:chat-42",
            user_id="wecom-user-7",
            # session_id intentionally omitted — the real gateway call site
            # (gateway/run.py::_set_session_env) never passes it either.
        )
        try:
            target = deployment["home"] / "config.yaml"
            raw = file_tools.read_file_tool(str(target))
            assert json.loads(raw)["error"] == READ_PATH_DENIED_CODE
        finally:
            clear_session_vars(tokens)

        row = _log_lines(deployment["home"])[0]
        assert row["session"] == "wecom:corpid-1:chat-42"
        assert row["subject"] == "wecom-user-7"
        assert row["platform"] == "wecom"
        assert "unknown" not in (row["session"], row["subject"], row["platform"])

    def test_path_not_allowed_reason_recorded(self, deployment, tmp_path):
        from tools import file_tools

        outside = tmp_path / "elsewhere.md"
        outside.write_text("nope\n")
        file_tools.read_file_tool(str(outside))
        row = _log_lines(deployment["home"])[0]
        assert row["reason"] == READGUARD_REASON_PATH
        assert row["tool"] == "read_file"

    def test_proc_environ_denied_before_device_guard(self, deployment):
        """/proc/*/environ must be refused by the read guard, not the
        stat-agnostic device-path guard.

        Regression for the ordering bug where ``read_file_tool`` ran the
        name-based device/special-file guard (``_is_blocked_device``, which
        also matches ``/proc/*/environ`` — see ``_is_blocked_device_path``)
        BEFORE ``_read_path_denied_response``. That guard returned its own
        ad hoc ``tool_error`` message instead of the structured
        ``path_not_allowed`` payload, and never called
        ``log_readguard_denial`` — so a credential-leaking read of
        ``/proc/self/environ`` produced neither the standard refusal shape
        nor an audit trail line.
        """
        from tools import file_tools

        raw = file_tools.read_file_tool("/proc/self/environ")
        payload = json.loads(raw)
        assert payload == DENIAL
        assert payload["error"] == READ_PATH_DENIED_CODE

        rows = _log_lines(deployment["home"])
        assert len(rows) == 1
        row = rows[0]
        assert row["tool"] == "read_file"
        assert row["reason"] == READGUARD_REASON_FILE
        assert row["path"] == os.path.normpath("/proc/self/environ")

    def test_search_files_denial_logged(self, deployment, tmp_path):
        from tools import file_tools

        outside = tmp_path / "elsewhere"
        outside.mkdir()
        file_tools.search_tool(pattern="x", path=str(outside))
        row = _log_lines(deployment["home"])[0]
        assert row["tool"] == "search_files"
        assert row["reason"] == READGUARD_REASON_PATH

    def test_terminal_denial_logged_with_offending_path(self, deployment):
        from tools import terminal_tool

        raw = terminal_tool.terminal_tool(
            command=f"cat {deployment['home']}/config.yaml"
        )
        assert json.loads(raw)["error"] == READ_PATH_DENIED_CODE
        rows = _log_lines(deployment["home"])
        assert rows and rows[-1]["tool"] == "terminal"
        assert rows[-1]["reason"] == READGUARD_REASON_FILE
        assert rows[-1]["path"].endswith("config.yaml")
        assert "supersecretvalue" not in json.dumps(rows[-1])

    def test_export_denial_logged(self, deployment):
        from tools import terminal_tool

        # ``workdir`` matters: bare ``czf`` is a relative operand to the read
        # guard, so a cwd outside the allowlist trips *that* guard first.
        raw = terminal_tool.terminal_tool(
            command=f"tar czf {deployment['out']}/kb.tgz {deployment['kb']}",
            workdir=str(deployment["out"]),
        )
        assert json.loads(raw)["error"] == KB_EXPORT_DENIED_CODE
        row = _log_lines(deployment["home"])[-1]
        assert row["tool"] == "terminal"
        assert row["reason"] == READGUARD_REASON_EXPORT

    def test_quota_denial_logged(self, deployment, monkeypatch):
        from tools import file_tools

        monkeypatch.setenv("HERMES_KB_READ_PER_TURN", "1")
        docs = deployment["kb"] / "canway-it-support"
        (docs / "b.md").write_text("second\n")
        file_tools.read_file_tool(str(docs / "a.md"))
        raw = file_tools.read_file_tool(str(docs / "b.md"))
        assert json.loads(raw)["error"] == KB_READ_QUOTA_CODE
        row = _log_lines(deployment["home"])[-1]
        assert row["tool"] == "read_file"
        assert row["reason"] == READGUARD_REASON_QUOTA

    def test_denial_survives_unwritable_log_dir(self, monkeypatch, tmp_path):
        from tools import file_tools

        home = tmp_path / "opt-data"
        home.mkdir()
        (home / "config.yaml").write_text("bot_secret: supersecretvalue\n")
        # ``logs`` is a regular FILE: makedirs and open both fail.
        (home / "logs").write_text("not a directory\n")
        monkeypatch.setenv("HERMES_HOME", str(home))

        raw = file_tools.read_file_tool(str(home / "config.yaml"))
        assert json.loads(raw) == {
            "error": READ_PATH_DENIED_CODE,
            "message": READ_PATH_DENIED_MESSAGE,
        }
        assert "supersecretvalue" not in raw


# ---------------------------------------------------------------------------
# 3. Bulk-export guard
# ---------------------------------------------------------------------------


class TestKbRootsResolution:
    def test_defaults_to_the_knowledge_read_root(self, deployment):
        assert get_kb_roots() == {os.path.realpath(deployment["kb"])}

    def test_explicit_env_wins(self, deployment, monkeypatch, tmp_path):
        other = tmp_path / "corpus"
        other.mkdir()
        monkeypatch.setenv("HERMES_KB_ROOTS", str(other))
        assert get_kb_roots() == {os.path.realpath(other)}

    def test_no_kb_root_means_no_export_guard(self, monkeypatch, tmp_path):
        monkeypatch.delenv("HERMES_READ_SAFE_ROOTS", raising=False)
        assert get_kb_roots() == set()
        assert get_command_export_denial("tar czf /tmp/x.tgz /srv", str(tmp_path)) is None


class TestExportForbidden:
    def _commands(self, kb: Path, out: Path) -> list[str]:
        return [
            f"tar czf {out}/kb.tgz {kb}",
            f"cp -r {kb}/canway-it-support {out}/",
            f"rsync -a {kb}/ {out}/",
            f"find {kb} -name '*.md' -exec cp {{}} {out} \\;",
            f"zip -r {out}/kb.zip {kb}/canway-it-support",
            f'python3 -c "import shutil;shutil.copytree(\'{kb}/canway-it-support\',\'{out}/x\')"',
        ]

    def test_all_export_shapes_refused(self, deployment):
        kb, out = deployment["kb"], deployment["out"]
        for command in self._commands(kb, out):
            denial = get_command_export_denial(command, str(out))
            assert denial is not None, command
            assert denial["error"] == KB_EXPORT_DENIED_CODE, command

    @pytest.mark.parametrize(
        "shape",
        [
            "cp -R {kb} {out}/",
            "cp -a {kb}/canway-it-support {out}/",
            "tar cf - {kb}",
            "7z a {out}/kb.7z {kb}",
            "find {kb} -type f | xargs cp -t {out}",
            "cat {kb}/canway-it-support/* > {out}/all.md",
            "scp -r {kb} user@host:/tmp",
        ],
    )
    def test_more_export_shapes_refused(self, deployment, shape):
        command = shape.format(kb=deployment["kb"], out=deployment["out"])
        denial = get_command_export_denial(command, str(deployment["out"]))
        assert denial is not None, command
        assert denial["error"] == KB_EXPORT_DENIED_CODE

    @pytest.mark.parametrize(
        "shape",
        [
            "cp {kb}/canway-it-support/a.md {out}/",
            "tar czf {out}/a.tgz {out}/report",
            "cat {kb}/canway-it-support/a.md",
            "grep -n line {kb}/canway-it-support/a.md",
        ],
    )
    def test_single_file_and_non_kb_shapes_allowed(self, deployment, shape):
        command = shape.format(kb=deployment["kb"], out=deployment["out"])
        assert get_command_export_denial(command, str(deployment["out"])) is None, command

    def test_terminal_returns_the_export_payload(self, deployment):
        from tools import terminal_tool

        raw = terminal_tool.terminal_tool(
            command=f"cp -r {deployment['kb']}/canway-it-support {deployment['out']}/"
        )
        payload = json.loads(raw)
        assert payload["error"] == KB_EXPORT_DENIED_CODE
        assert payload["message"] == "知识库内容不提供整包导出"
        assert payload["status"] == "blocked"

    def test_terminal_allows_single_file_copy(self, deployment):
        from tools import terminal_tool

        raw = terminal_tool.terminal_tool(
            command=(
                f"cp {deployment['kb']}/canway-it-support/a.md {deployment['out']}/"
            )
        )
        payload = json.loads(raw)
        assert payload.get("error") not in (
            KB_EXPORT_DENIED_CODE,
            READ_PATH_DENIED_CODE,
        )


# ---------------------------------------------------------------------------
# 4. Per-turn knowledge read quota
# ---------------------------------------------------------------------------


class TestKbReadQuota:
    def _kb_docs(self, kb: Path, count: int) -> list[Path]:
        docs = []
        for i in range(count):
            doc = kb / "canway-it-support" / f"doc-{i}.md"
            doc.write_text(f"knowledge body {i}\n")
            docs.append(doc)
        return docs

    def test_twenty_first_kb_read_refused(self, deployment):
        from tools import file_tools

        docs = self._kb_docs(deployment["kb"], 21)
        for doc in docs[:20]:
            raw = file_tools.read_file_tool(str(doc))
            assert json.loads(raw).get("error") != KB_READ_QUOTA_CODE

        raw = file_tools.read_file_tool(str(docs[20]))
        payload = json.loads(raw)
        assert payload["error"] == KB_READ_QUOTA_CODE
        assert "20" in payload["message"]
        # The refusal steers to retrieval and leaks no content.
        assert "knowledge body" not in raw

    def test_new_turn_resets_the_counter(self, deployment, monkeypatch):
        from tools import file_tools

        monkeypatch.setenv("HERMES_KB_READ_PER_TURN", "2")
        docs = self._kb_docs(deployment["kb"], 4)
        file_tools.read_file_tool(str(docs[0]))
        file_tools.read_file_tool(str(docs[1]))
        assert json.loads(file_tools.read_file_tool(str(docs[2])))["error"] == (
            KB_READ_QUOTA_CODE
        )

        reset_kb_read_quota()  # turn boundary (agent/turn_context.py)
        assert json.loads(file_tools.read_file_tool(str(docs[3]))).get("error") != (
            KB_READ_QUOTA_CODE
        )

    def test_non_kb_reads_are_not_counted(self, deployment, monkeypatch):
        from tools import file_tools

        monkeypatch.setenv("HERMES_KB_READ_PER_TURN", "2")
        skills = deployment["home"] / "skills" / "x"
        for i in range(5):
            doc = skills / f"note-{i}.md"
            doc.write_text(f"skill note {i}\n")
            raw = file_tools.read_file_tool(str(doc))
            assert json.loads(raw).get("error") != KB_READ_QUOTA_CODE

        docs = self._kb_docs(deployment["kb"], 2)
        for doc in docs:
            raw = file_tools.read_file_tool(str(doc))
            assert json.loads(raw).get("error") != KB_READ_QUOTA_CODE

    def test_search_files_is_not_counted(self, deployment, monkeypatch):
        from tools import file_tools

        monkeypatch.setenv("HERMES_KB_READ_PER_TURN", "1")
        for _ in range(3):
            file_tools.search_tool(pattern="kb", path=str(deployment["kb"]))
        docs = self._kb_docs(deployment["kb"], 1)
        raw = file_tools.read_file_tool(str(docs[0]))
        assert json.loads(raw).get("error") != KB_READ_QUOTA_CODE

    def test_bypass_lifts_the_quota(self, deployment, monkeypatch):
        monkeypatch.setenv("HERMES_KB_READ_PER_TURN", "1")
        monkeypatch.setenv("HERMES_READ_SAFE_ROOTS_BYPASS", "1")
        doc = deployment["kb"] / "canway-it-support" / "a.md"
        for _ in range(5):
            assert check_kb_read_quota(str(doc)) is None

    def test_quota_is_per_session(self, deployment, monkeypatch):
        from gateway.session_context import clear_session_vars, set_session_vars

        monkeypatch.setenv("HERMES_KB_READ_PER_TURN", "1")
        doc = deployment["kb"] / "canway-it-support" / "a.md"

        tokens = set_session_vars(session_id="sess-a")
        try:
            assert check_kb_read_quota(str(doc)) is None
            assert check_kb_read_quota(str(doc))["error"] == KB_READ_QUOTA_CODE
        finally:
            clear_session_vars(tokens)

        tokens = set_session_vars(session_id="sess-b")
        try:
            assert check_kb_read_quota(str(doc)) is None
        finally:
            clear_session_vars(tokens)
