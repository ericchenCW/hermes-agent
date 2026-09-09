"""Shared file safety rules used by both tools and ACP shims.

Every guard here is defense-in-depth, NOT a security boundary: the terminal
tool runs as the same OS user and can read/write anything. The value is a
clear denial for models that respect tool errors plus a visible audit trail.
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import suppress
from typing import Optional


def _constants_path(getter_name: str) -> Path:
    """Call ``hermes_constants.<getter_name>()`` (local import avoids cycles); ``~/.hermes`` on any failure."""
    try:
        import hermes_constants

        return getattr(hermes_constants, getter_name)()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def _hermes_home_path() -> Path:
    """Active HERMES_HOME (profile-aware). Tests monkeypatch this name."""
    return _constants_path("get_hermes_home")


def _hermes_root_path() -> Path:
    """Hermes root dir (parent of any profile, never per-profile)."""
    return _constants_path("get_default_hermes_root")


def _hermes_dirs() -> list[Path]:
    """Resolved active HERMES_HOME and global root, deduplicated.

    Both are checked so credential stores at <root>/... stay guarded when
    running under a profile (HERMES_HOME = <root>/profiles/<name>).
    """
    return list(dict.fromkeys(_resolve_each((_hermes_home_path(), _hermes_root_path()))))


def _resolve_each(paths) -> list[Path]:
    """``p.resolve()`` for each path, skipping ones that fail to resolve."""
    out: list[Path] = []
    for p in paths:
        with suppress(Exception):
            out.append(p.resolve())
    return out


def _is_under(resolved: str | Path, base: str | Path) -> bool:
    """True when ``resolved`` equals ``base`` or lies below it (both already resolved);
    ``Path`` inputs use ``relative_to`` (platform semantics), ``str`` a realpath prefix test."""
    if isinstance(resolved, Path):
        try:
            return resolved.relative_to(base) is not None
        except ValueError:
            return False
    return resolved == base or resolved.startswith(str(base) + os.sep)


def _resolve_target(path: str) -> Optional[Path]:
    """``Path(expanduser(path)).resolve()``, or None when resolution fails."""
    with suppress(OSError, RuntimeError):
        return Path(os.path.expanduser(str(path))).resolve()
    return None


def _home_and_resolved(path: str) -> tuple[str, str]:
    """``(realpath(~), realpath(expanduser(path)))`` — the write-guard coordinate pair."""
    return tuple(os.path.realpath(os.path.expanduser(p)) for p in ("~", str(path)))


def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    # ``~/.ssh/config`` is deliberately NOT hard-denied: no key bytes, and editing
    # it is routine. It can carry ProxyCommand / Match exec, so it goes through the
    # approval gate instead (build_write_approval_paths).
    home_files = (
        (".ssh", "authorized_keys"), (".ssh", "id_rsa"), (".ssh", "id_ed25519"),
        (".netrc",), (".pgpass",), (".npmrc",), (".pypirc",), (".git-credentials",),
    )
    # Both the active-profile and top-level copies: overwriting the root .env leaks
    # credentials across every profile that inherits from it; the root Anthropic
    # PKCE store is still read by default/non-profile sessions when a profile is
    # active; bws_cache.enc.json is the Bitwarden Secrets Manager encrypted cache.
    hermes_files = (".env", ".anthropic_oauth.json", os.path.join("cache", "bws_cache.enc.json"))
    paths = [
        *(os.path.join(home, *f) for f in home_files),
        *(str(base / f) for f in hermes_files for base in (_hermes_home_path(), _hermes_root_path())),
        "/etc/sudoers", "/etc/passwd", "/etc/shadow",
    ]
    return {os.path.realpath(p) for p in paths}


def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    paths = [
        *(os.path.join(home, d) for d in (".ssh", ".aws", ".gnupg", ".kube")),
        "/etc/sudoers.d", "/etc/systemd",
        *(os.path.join(home, *d) for d in ((".docker",), (".azure",), (".config", "gh"), (".config", "gcloud"))),
    ]
    return [os.path.realpath(p) + os.sep for p in paths]


def get_safe_write_roots() -> set[str]:
    """Resolved HERMES_WRITE_SAFE_ROOT paths (``os.pathsep``-separated list)."""
    roots: set[str] = set()
    for path in filter(None, os.getenv("HERMES_WRITE_SAFE_ROOT", "").split(os.pathsep)):
        with suppress(OSError, ValueError):
            roots.add(os.path.realpath(os.path.expanduser(path)))
    return roots


def build_write_approval_paths(home: str) -> set[str]:
    """Paths that need human APPROVAL to write but are not hard-denied credentials.

    ``~/.ssh/config`` is routine to edit and holds no key bytes, but can carry
    ``ProxyCommand`` / ``Match exec``. Interactive file tools prompt
    (approve-once/session/always, like the terminal tool's ``~/.ssh`` gate);
    non-interactive callers (ACP shims, background jobs) fail closed.
    """
    return {os.path.realpath(os.path.join(home, ".ssh", "config"))}


# HERMES_HOME / root subpaths that the agent's generic file tools must not
# rewrite. Session transcripts (state.db, sessions/) are application-owned
# state whose rewrite can falsify history and break resume/compression;
# mcp-tokens/ and pairing/ hold credential material.
_HERMES_PROTECTED_SUBPATHS = ("state.db", "sessions", "mcp-tokens", "pairing")


def _classify_write_denial(path: str) -> Optional[str]:
    """Return ``'credential'``, ``'safe_root'``, or ``None`` if writes are allowed."""
    home, resolved = _home_and_resolved(path)

    # Approval-gated paths are allowed at this layer so interactive tools can
    # prompt; checked first so the ``.ssh/`` prefix deny doesn't swallow them.
    if resolved in build_write_approval_paths(home):
        return None

    if resolved in build_write_denied_paths(home) or any(
        resolved.startswith(prefix) for prefix in build_write_denied_prefixes(home)
    ):
        return "credential"

    for base in _hermes_dirs():
        for sub in _HERMES_PROTECTED_SUBPATHS:
            with suppress(Exception):
                if _is_under(resolved, os.path.realpath(os.path.join(str(base), sub))):
                    return "credential"

    safe_roots = get_safe_write_roots()
    if safe_roots and not any(_is_under(resolved, root) for root in safe_roots):
        return "safe_root"

    return None


def is_write_denied(path: str) -> bool:
    """Return True if path is blocked by the write denylist or safe root."""
    return _classify_write_denial(path) is not None


def get_write_denied_error(path: str, *, verb: str = "Write") -> Optional[str]:
    """Return a user/model-facing error when writes to ``path`` are blocked."""
    denial = _classify_write_denial(path)
    if denial == "safe_root":
        roots_display = os.pathsep.join(sorted(get_safe_write_roots()))
        return (
            f"{verb} denied: '{path}' is outside HERMES_WRITE_SAFE_ROOT "
            f"({roots_display}). Unset the variable or add this path's directory prefix."
        )
    return f"{verb} denied: '{path}' is a protected system/credential file." if denial else None


def is_write_approval_required(path: str) -> bool:
    """True if ``path`` is approval-gated (``~/.ssh/config``): interactive callers
    prompt, callers without a channel treat it as a block (fail closed)."""
    home, resolved = _home_and_resolved(path)
    return resolved in build_write_approval_paths(home)


# Secret-bearing project-local env file basenames, blocked anywhere on disk.
_BLOCKED_PROJECT_ENV_BASENAMES: set[str] = {
    ".env", ".env.local", ".env.development", ".env.production", ".env.test", ".env.staging", ".envrc",
}

_DID_SUFFIX = (
    " (Defense-in-depth — not a security boundary; the terminal tool can still bypass.)"
)

# Exact-file credential stores under HERMES_HOME / <root>. The agent never
# needs these directly — provider tools consume them through internal channels.
# bws_cache.json is the Bitwarden Secrets Manager disk cache: plaintext secret values.
_CREDENTIAL_FILE_NAMES = (
    "auth.json", "auth.lock", ".anthropic_oauth.json", ".env", "webhook_subscriptions.json",
    os.path.join("auth", "google_oauth.json"), os.path.join("cache", "bws_cache.json"),
)

# Directory-prefix read denies under HERMES_HOME / <root>: (subdir, message for
# the directory itself, message for a file inside). browser-profile/ is a copy
# of the user's Cookies / Login Data — the same credential class as auth.json.
_READ_DENIED_DIRS = (
    ("mcp-tokens",
     "is the Hermes MCP token directory and cannot be read directly.",
     "is a Hermes MCP token file and cannot be read directly."),
    ("browser-profile",
     "is the Hermes real-profile browser snapshot directory (copied cookies/logins) and cannot be read directly.",
     "is inside the Hermes real-profile browser snapshot (copied cookies/logins) and cannot be read directly."),
)


def get_read_block_error(path: str) -> Optional[str]:
    """Return an error message when a read targets a denied Hermes path.

    Blocked: internal skill-hub caches (prompt-injection carriers), credential
    stores under HERMES_HOME and the global root (exact files, plus anything
    under ``mcp-tokens/`` and ``browser-profile/``), and project-local ``.env``
    files anywhere on disk (``.env.example`` is the documented-shape substitute).

    Callers that resolve relative paths against a non-process cwd (e.g.
    ``TERMINAL_CWD``) MUST pass an absolute path: ``resolve()`` here anchors at
    the process cwd, so a relative ``"auth.json"`` would miss the denylist.
    """
    resolved = Path(path).expanduser().resolve()
    hermes_dirs = _hermes_dirs()
    reason = None
    if any(_is_under(resolved, hd / "skills" / ".hub") for hd in hermes_dirs):
        reason = (
            "is an internal Hermes cache file and cannot be read directly to prevent "
            "prompt injection. Use the skills_list or skill_view tools instead."
        )
    elif any(resolved in _resolve_each(hd / name for hd in hermes_dirs) for name in _CREDENTIAL_FILE_NAMES):
        reason = (
            "is a Hermes credential store and cannot be read directly. Provider tools "
            "consume these credentials through internal channels." + _DID_SUFFIX
        )
    else:
        for subdir, dir_msg, file_msg in _READ_DENIED_DIRS:
            for blocked_dir in _resolve_each(hd / subdir for hd in hermes_dirs):
                if _is_under(resolved, blocked_dir):
                    reason = (dir_msg if resolved == blocked_dir else file_msg) + _DID_SUFFIX
                    break
            if reason:
                break
        if reason is None and resolved.name.lower() in _BLOCKED_PROJECT_ENV_BASENAMES:
            reason = (
                "is a secret-bearing environment file and cannot be read to prevent credential "
                "leakage. If you need to check the file structure, read .env.example instead." + _DID_SUFFIX
            )
    return f"Access denied: {path} {reason}" if reason else None


def raise_if_read_blocked(path: str) -> None:
    """Raise ``ValueError`` if ``path`` is a denied Hermes read (see ``get_read_block_error``).

    Shared chokepoint for provider input-loading sites (e.g. image-gen local
    paths). Best-effort: unexpected internal errors no-op rather than break
    local-file loading; a real block still propagates.
    """
    try:
        blocked = get_read_block_error(path)
    except Exception:  # noqa: BLE001 - guard must never break local-file loading
        return
    if blocked:
        raise ValueError(blocked)


def _resolve_active_profile_name() -> str:
    """Active profile name from HERMES_HOME: ``~/.hermes`` -> ``"default"``,
    ``~/.hermes/profiles/X`` -> ``"X"``; ``"default"`` on any resolution failure."""
    try:
        parts = _hermes_home_path().resolve().relative_to(_hermes_root_path().resolve() / "profiles").parts
    except (OSError, RuntimeError, ValueError):
        return "default"
    return parts[0] if parts else "default"


# --- Sandbox-mirror write guard ---
# Non-local terminal backends bind a sandbox-local dir to the container's $HOME:
#   <HERMES_HOME>/profiles/<name>/sandboxes/<backend>/<task>/home/.hermes/...
# A host-side write there lands on a mirror the host never reads: silent success,
# divergent copies. Path-shape-only detection, independent of the active profile;
# the inner-container case (bind mount strips the prefix) is classify_container_mirror_target.

_SANDBOX_MIRROR_WARNING = (
    "Sandbox-mirror write blocked by soft guard: {target_path} "
    "sits under {mirror_root!r}, which is {body} "
    "Use the host-side tool for authoritative state (e.g. ``memory`` for memories), "
    "or address the host path directly. To bypass {bypass} with ``cross_profile=True``. "
    "(Defense-in-depth — not a security boundary; the terminal tool can still bypass.)"
)


def _mirror_info(target: Path, mirror_root: Path, inner_path: str) -> dict:
    """Common ``classify_*_mirror_target`` result shape."""
    return {"target_path": str(target), "mirror_root": str(mirror_root), "inner_path": inner_path}


def classify_sandbox_mirror_target(path: str) -> Optional[dict]:
    """Classify a write target as a sandbox-mirror of authoritative Hermes state: ``None``
    for non-mirror paths, else ``target_path`` (resolved), ``mirror_root`` (the
    ``…/home/.hermes`` prefix) and ``inner_path`` (what the agent meant on the host)."""
    target = _resolve_target(path)
    parts = target.parts if target is not None else ()
    # Need at least: sandboxes / <backend> / <task> / home / .hermes / <thing>; inner_idx = the .hermes part.
    inner_idx = next(
        (i + 4 for i, part in enumerate(parts)
         if part == "sandboxes" and i + 5 < len(parts) and parts[i + 3] == "home" and parts[i + 4] == ".hermes"),
        None,
    )
    if inner_idx is None:
        return None
    inner = str(Path(*parts[inner_idx + 1:])) if inner_idx + 1 < len(parts) else ""
    return _mirror_info(target, Path(*parts[: inner_idx + 1]), inner)


def _mirror_warning(info: Optional[dict], body: str, bypass: str) -> Optional[str]:
    """Render ``_SANDBOX_MIRROR_WARNING`` for a classify_* result (``body`` may use ``{inner_path}``)."""
    if info is None:
        return None
    return _SANDBOX_MIRROR_WARNING.format(**info, body=body.format(inner_path=info["inner_path"]), bypass=bypass)


def get_sandbox_mirror_warning(path: str) -> Optional[str]:
    """Model-facing soft-guard warning when ``path`` lands in a sandbox mirror, else ``None``;
    the caller surfaces it as a tool-result error and ``cross_profile=True`` bypasses."""
    return _mirror_warning(
        classify_sandbox_mirror_target(path),
        "a per-task mirror created by a non-local terminal backend (docker/daytona/etc.). "
        "Writes here land on a copy that the host Hermes process never reads — the "
        "authoritative file is likely {inner_path!r} under the real HERMES_HOME.",
        "this guard after explicit user direction, retry the call",
    )


def classify_container_mirror_target(path: str, mirror_prefix: str | None = None) -> Optional[dict]:
    """Classify a write target as a container-side sandbox mirror. Inside the container
    the bind mount strips the ``sandboxes/`` prefix (the agent sees plain ``/root/.hermes/…``),
    so the caller supplies ``mirror_prefix`` once it knows file tools run in a docker sandbox.
    ``None`` without a prefix or outside it, else ``target_path``/``mirror_root``/``inner_path``."""
    target, mirror = _resolve_target(path), _resolve_target(mirror_prefix) if mirror_prefix else None
    if target is None or mirror is None or not _is_under(target, mirror):
        return None
    return _mirror_info(target, mirror, target.relative_to(mirror).as_posix())


def get_container_mirror_warning(path: str, mirror_prefix: str | None = None) -> Optional[str]:
    """Model-facing soft-guard warning when ``path`` lands in the container's mirror, else ``None``."""
    return _mirror_warning(
        classify_container_mirror_target(path, mirror_prefix),
        "the container's bind-mounted home — a per-task mirror that the host Hermes "
        "process never reads. The authoritative file is {inner_path!r} under "
        "the real HERMES_HOME.",
        "after explicit user direction, retry",
    )




# ---------------------------------------------------------------------------
# Read path allowlist (HERMES_READ_SAFE_ROOTS)
#
# Sibling of HERMES_WRITE_SAFE_ROOT, for the READ direction. Motivation: a
# Hermes bot exposed to untrusted chat users through a read-only toolset
# (``read_file`` / ``search_files`` / ``terminal``) could be talked into
# returning its own ``config.yaml`` — model endpoints, runtime token
# references, IM credential references, internal addresses.
#
# Two layers, both enforced by :func:`get_read_path_denial`:
#
#   1. **Allowlist** — when ``HERMES_READ_SAFE_ROOTS`` is set, a read target
#      must resolve inside one of the listed roots. Anything else is refused
#      with a fixed, content-free payload that leaks neither the file's
#      existence nor its contents.
#   2. **Denylist** — always on (it stacks on top of the allowlist): Hermes
#      home/root trees, ``/opt/data/config.yaml``, any ``.env*`` file,
#      ``*.key`` / ``*.pem``, ``/proc`` and ``/etc``.
#
# Unlike :func:`get_read_block_error` (documented as defense-in-depth, not a
# boundary) this guard also covers the terminal tool's own read commands, so
# a ``cat`` fallback does not walk around it.
#
# ``HERMES_READ_SAFE_ROOTS_BYPASS=1`` disables BOTH layers. It exists for the
# maintainer / built-in operator role that administers the deployment itself.
# ---------------------------------------------------------------------------

READ_SAFE_ROOTS_ENV = "HERMES_READ_SAFE_ROOTS"
READ_SAFE_ROOTS_BYPASS_ENV = "HERMES_READ_SAFE_ROOTS_BYPASS"

#: Structured refusal returned for every read-path denial. Deliberately
#: uniform: the caller learns nothing about the target beyond "not allowed"
#: — not whether it exists, not what it contains, not which rule fired.
READ_PATH_DENIED_CODE = "path_not_allowed"
READ_PATH_DENIED_MESSAGE = "该路径不在允许读取的范围内"

_TRUTHY = {"1", "true", "yes", "on"}

# Suffixes that always hold key material.
_READ_DENIED_SUFFIXES = (".key", ".pem")

# Absolute trees that are never readable through the managed tools.
_READ_DENIED_SYSTEM_PREFIXES = ("/proc", "/etc")

# Exact absolute files that are never readable (beyond the Hermes trees).
_READ_DENIED_EXACT = ("/opt/data/config.yaml",)


def is_read_safe_root_bypassed() -> bool:
    """Return True when ``HERMES_READ_SAFE_ROOTS_BYPASS`` disables the guard.

    Config-key equivalent: ``security.read_safe_roots_bypass: true`` is
    bridged to this env var by the CLI/gateway startup, the same way
    ``security.redact_secrets`` bridges to ``HERMES_REDACT_SECRETS``.
    """
    return os.getenv(READ_SAFE_ROOTS_BYPASS_ENV, "").strip().lower() in _TRUTHY


def get_safe_read_roots() -> set[str]:
    """Return resolved ``HERMES_READ_SAFE_ROOTS`` paths.

    Accepts commas (the documented separator) as well as ``os.pathsep``, so
    ``/knowledge,/opt/data/kb`` and ``/knowledge:/opt/data/kb`` both work.
    Empty/unresolvable entries are dropped; an empty result means "no
    allowlist configured" (denylist only).
    """
    env = os.getenv(READ_SAFE_ROOTS_ENV, "")
    if not env:
        return set()
    raw: list[str] = []
    for chunk in env.split(os.pathsep):
        raw.extend(chunk.split(","))
    roots: set[str] = set()
    for path in raw:
        path = path.strip()
        if not path:
            continue
        try:
            roots.add(os.path.realpath(os.path.expanduser(path)))
        except (OSError, ValueError):
            continue
    return roots


def _resolve_for_read_guard(path: str) -> Optional[str]:
    """realpath+expanduser a candidate path; None when it cannot be resolved.

    ``os.path.realpath`` resolves symlinks and normalizes ``..`` without
    touching the filesystem for missing components, so a symlink pointing
    out of an allowed root and a ``../../etc/passwd`` traversal both land on
    their true target before the prefix test runs.
    """
    try:
        return os.path.realpath(os.path.expanduser(str(path)))
    except (OSError, ValueError):
        return None


def _under(candidate: str, prefix: str) -> bool:
    """True when ``candidate`` is ``prefix`` itself or lives beneath it."""
    return candidate == prefix or candidate.startswith(prefix.rstrip(os.sep) + os.sep)


def _denied_prefixes() -> list[str]:
    """Always-denied directory trees, in both raw and symlink-resolved form.

    macOS ships ``/etc`` and ``/tmp`` as symlinks into ``/private``, so a
    realpath-only comparison would miss ``/etc/passwd`` (it resolves to
    ``/private/etc/passwd``). Listing both spellings keeps the denylist
    platform-independent.
    """
    prefixes: list[str] = []
    for prefix in _READ_DENIED_SYSTEM_PREFIXES:
        if prefix not in prefixes:
            prefixes.append(prefix)
        try:
            real = os.path.realpath(prefix)
        except (OSError, ValueError):
            continue
        if real not in prefixes:
            prefixes.append(real)
    # $HERMES_HOME (profile-aware), the global Hermes root, and the literal
    # ~/.hermes default — the last one matters when HERMES_HOME has been
    # pointed elsewhere but the stock profile still exists on disk.
    for base in (
        _hermes_home_path(),
        _hermes_root_path(),
        os.path.expanduser("~/.hermes"),
    ):
        for form in (str(base), os.path.realpath(base)):
            if form and form not in prefixes:
                prefixes.append(form)
    return prefixes


def _read_denylist_hit(candidates: tuple[str, ...]) -> bool:
    """Return True when any spelling of the target hits the read denylist."""
    prefixes = _denied_prefixes()
    for candidate in candidates:
        if not candidate:
            continue
        lowered = os.path.basename(candidate).lower()

        # Any .env / .env.local / .envrc / .env.whatever anywhere on disk.
        if lowered.startswith(".env"):
            return True

        # Key material by extension.
        if lowered.endswith(_READ_DENIED_SUFFIXES):
            return True

        if candidate in _READ_DENIED_EXACT:
            return True

        for prefix in prefixes:
            try:
                if _under(candidate, prefix):
                    return True
            except (OSError, ValueError):
                continue

    return False


def get_read_path_denial(path: str) -> Optional[dict]:
    """Return the structured refusal for ``path``, or ``None`` when allowed.

    The returned mapping is exactly what tools should serialize back to the
    model::

        {"error": "path_not_allowed", "message": "该路径不在允许读取的范围内"}

    It is intentionally identical for every denial reason so the response
    cannot be used as an oracle for path existence, file type, or which rule
    fired.

    Callers that resolve relative paths against a non-process cwd (the file
    tools' ``TERMINAL_CWD``) MUST pass the already-absolute path — this
    function's own ``realpath`` is anchored at the Python process cwd.
    """
    if is_read_safe_root_bypassed():
        return None

    resolved = _resolve_for_read_guard(path)
    if resolved is None:
        # Unresolvable input: fail closed.
        return {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}

    # The denylist is checked against BOTH the symlink-resolved path and the
    # merely-normalized one, so neither ``/etc/passwd`` (a symlink on macOS)
    # nor a symlink pointing INTO a denied tree can slip through.
    try:
        normalized = os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))
    except (OSError, ValueError):
        normalized = resolved
    if _read_denylist_hit((resolved, normalized)):
        return {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}

    roots = get_safe_read_roots()
    if roots:
        for root in roots:
            if resolved == root or resolved.startswith(root + os.sep):
                return None
        return {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}

    return None


def is_read_path_denied(path: str) -> bool:
    """Boolean form of :func:`get_read_path_denial`."""
    return get_read_path_denial(path) is not None


# ---------------------------------------------------------------------------
# Terminal-command read guard
#
# The file tools' allowlist is worthless if ``cat /opt/data/config.yaml``
# walks around it, so the same rules are applied to path operands extracted
# from shell commands. Two tiers:
#
#   * **Known read commands** (``cat``/``head``/``grep``/``find``/``ls``/…):
#     every operand is treated as a path and fully validated.
#   * **Everything else, including commands we cannot parse**: any token that
#     is an absolute path, a ``~`` path, or contains a ``..`` traversal is
#     validated. Per the brief, a command we cannot fully resolve is refused
#     as soon as it mentions an out-of-allowlist absolute path or a ``..``
#     escape, rather than being waved through.
#
# This is a *path* guard, not a shell emulator: it does not try to model
# every redirection or expansion. It raises the cost of the obvious
# exfiltration shapes and fails closed on the ones it cannot read.
# ---------------------------------------------------------------------------

import re as _re
import shlex as _shlex

#: Commands whose operands are file paths.
_READ_COMMANDS = frozenset({
    "cat", "bat", "tac", "nl", "head", "tail", "less", "more", "most",
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "zgrep",
    "find", "fd", "ls", "ll", "dir", "tree", "stat", "file", "du", "wc",
    "sed", "awk", "gawk", "mawk", "cut", "sort", "uniq", "tr", "column",
    "od", "xxd", "hexdump", "strings", "base64", "md5sum", "sha256sum",
    "diff", "cmp", "readlink", "realpath", "jq", "yq", "xmllint",
    "python", "python3", "perl", "ruby", "node", "php",
    "cp", "install", "rsync", "scp", "tar", "zip", "unzip", "gzip", "gunzip",
    "openssl", "curl", "wget", "dd", "vi", "vim", "nano", "emacs", "view",
})

#: Read commands whose FIRST non-flag operand is a script/pattern, not a
#: path (``sed -n '1,5p' f``, ``grep foo f``, ``python -c 'code'``). That
#: operand is only checked when it is itself absolute or ``~``-rooted.
_SCRIPT_FIRST_COMMANDS = frozenset({
    "grep", "egrep", "fgrep", "rg", "ag", "ack", "zgrep",
    "sed", "awk", "gawk", "mawk", "perl", "ruby", "python", "python3",
    "node", "php",
})

#: Flags that mean "the pattern/script came from a flag", so the first
#: operand is a path after all.
_PATTERN_FLAGS = frozenset({"-e", "-f", "--regexp", "--file", "--expression", "-c"})

#: Wrapper commands that prefix the real command.
_COMMAND_PREFIXES = frozenset({
    "sudo", "doas", "env", "command", "builtin", "exec", "nohup", "time",
    "timeout", "nice", "ionice", "stdbuf", "xargs", "watch", "strace",
})

_SEGMENT_SPLIT_RE = _re.compile(r"&&|\|\||\$\(|[;\n|&()`]")

_ENV_ASSIGN_TOKEN_RE = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

#: Absolute paths embedded INSIDE another token — ``python -c 'open("/opt/…")'``,
#: ``awk '{…}' /etc/x``, a heredoc body, an unparsed quoted blob. The
#: lookbehind stops ``s/foo/bar/`` (sed script) and ``http://…`` from matching,
#: because their ``/`` follows a word character.
_EMBEDDED_ABS_PATH_RE = _re.compile(r"(?<![\w:])(/[A-Za-z0-9._+\-]+(?:/[A-Za-z0-9._+\-]+)*)")


def _embedded_paths(token: str) -> list[str]:
    """Absolute-path substrings hiding inside a single shell token."""
    if not token or "://" in token:
        return []
    return _EMBEDDED_ABS_PATH_RE.findall(token)


def _looks_absolute_or_escape(token: str) -> bool:
    """True when a token names an absolute path or escapes upward."""
    if not token or "://" in token:
        return False
    if token.startswith("/") or token.startswith("~"):
        return True
    parts = token.replace("\\", "/").split("/")
    return ".." in parts


def _command_path_candidates(command: str) -> list[str]:
    """Extract path-ish operands from a shell command string."""
    candidates: list[str] = []

    # Baseline pass over the raw string: any multi-segment absolute path
    # mentioned ANYWHERE in the command is checked, whatever the shell
    # structure around it. This is what catches shapes the tokenizer cannot
    # model — ``python -c 'open("/opt/data/config.yaml")'``, heredocs,
    # nested substitutions. Single-segment matches (``/etc`` on its own) are
    # left to the per-command pass so an incidental mention in a commit
    # message or a comment does not fail the whole command closed.
    for token in (command or "").split():
        for found in _embedded_paths(token.strip("'\"")):
            if found.count("/") >= 2:
                candidates.append(found)

    for segment in _SEGMENT_SPLIT_RE.split(command or ""):
        segment = segment.strip()
        if not segment:
            continue
        try:
            argv = _shlex.split(segment, posix=True)
        except ValueError:
            # Unbalanced quotes — we cannot parse it. Fall back to the
            # conservative tier: flag anything that looks like an absolute
            # path or a traversal.
            for token in segment.split():
                token = token.strip("'\"")
                if _looks_absolute_or_escape(token):
                    candidates.append(token)
                candidates.extend(_embedded_paths(token))
            continue
        # Strip leading env assignments and wrapper commands.
        while argv and (
            _ENV_ASSIGN_TOKEN_RE.match(argv[0])
            or os.path.basename(argv[0]) in _COMMAND_PREFIXES
        ):
            argv = argv[1:]
        if not argv:
            continue
        name = os.path.basename(argv[0])
        rest = argv[1:]
        is_read_cmd = name in _READ_COMMANDS
        skip_first_operand = (
            is_read_cmd
            and name in _SCRIPT_FIRST_COMMANDS
            and not any(flag in _PATTERN_FLAGS for flag in rest)
        )
        seen_operand = False
        for token in rest:
            if not token or token == "-":
                continue
            # An absolute path hiding inside a bigger token (a ``python -c``
            # program, an ``awk`` body, a quoted blob) is checked no matter
            # which command it belongs to.
            candidates.extend(_embedded_paths(token))
            if token.startswith("-"):
                continue
            if not is_read_cmd:
                if _looks_absolute_or_escape(token):
                    candidates.append(token)
                continue
            if skip_first_operand and not seen_operand:
                seen_operand = True
                # A pattern/script operand is only a path when it is spelled
                # as one (``grep /etc/passwd`` is still worth refusing).
                if token.startswith("/") or token.startswith("~"):
                    candidates.append(token)
                continue
            seen_operand = True
            candidates.append(token)
    return candidates


def get_command_read_denial(command: str, cwd: Optional[str] = None) -> Optional[dict]:
    """Return the structured refusal when a shell command reads a blocked path.

    ``cwd`` anchors relative operands (the terminal tool's resolved working
    directory). Returns ``None`` when nothing in the command is out of
    bounds, or when ``HERMES_READ_SAFE_ROOTS_BYPASS`` is set.
    """
    if is_read_safe_root_bypassed():
        return None
    if not command or not isinstance(command, str):
        return None

    base = cwd or os.getcwd()
    for token in _command_path_candidates(command):
        expanded = os.path.expanduser(token)
        if not os.path.isabs(expanded):
            try:
                expanded = os.path.join(base, expanded)
            except (OSError, ValueError, TypeError):
                continue
        denial = get_read_path_denial(expanded)
        if denial is not None:
            return denial
    return None


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

PROFILE_SCOPED_AREAS = ("skills", "plugins", "cron", "memories")

def classify_cross_profile_target(path: str) -> Optional[dict]:
    """Classify a write target as cross-profile if it lands in another
    profile's scoped area (skills/plugins/cron/memories).

    Returns ``None`` when the target is outside Hermes scope, or is inside
    the ACTIVE profile, or doesn't hit a profile-scoped area. Otherwise
    returns a dict with:

      * ``active_profile``: name of the profile the agent is running as
      * ``target_profile``: name of the profile the path belongs to
      * ``area``: which scoped area (``"skills"``, ``"plugins"``, etc.)
      * ``target_path``: the resolved path string

    The caller decides what to do with the result — surface a warning to
    the model, prompt the user, or (with explicit consent /
    ``cross_profile=True``) proceed anyway.
    """
    try:
        target = Path(os.path.expanduser(str(path))).resolve()
        root_real = _hermes_root_path().resolve()
    except (OSError, RuntimeError):
        return None

    target_profile: Optional[str] = None
    area: Optional[str] = None

    try:
        rel = target.relative_to(root_real)
    except ValueError:
        return None

    parts = rel.parts
    if not parts:
        return None

    if parts[0] in PROFILE_SCOPED_AREAS:
        # ``<root>/<area>/...`` → default profile.
        target_profile = "default"
        area = parts[0]
    elif (
        parts[0] == "profiles"
        and len(parts) >= 3
        and parts[2] in PROFILE_SCOPED_AREAS
    ):
        # ``<root>/profiles/<name>/<area>/...`` → named profile.
        target_profile = parts[1]
        area = parts[2]
    else:
        return None

    active_profile = _resolve_active_profile_name()
    if target_profile == active_profile:
        # In-profile write — not a cross-profile event.
        return None

    return {
        "active_profile": active_profile,
        "target_profile": target_profile,
        "area": area,
        "target_path": str(target),
    }

def get_cross_profile_warning(path: str) -> Optional[str]:
    """RETIRED (maintainer decision): always returns ``None``.

    The cross-profile write guard was removed — profiles were never
    isolated (same OS user; the terminal tool writes anywhere), so the
    block was ceremony that cost every schema real tokens and taught a
    bypass arg. The system prompt's active-profile hint remains the only
    steering; the classifier below survives for that hint and for
    diagnostics. Kept as a stub so external callers/plugins fail soft.
    """
    return None
# ---- END PLUGIN-COMPAT ----
