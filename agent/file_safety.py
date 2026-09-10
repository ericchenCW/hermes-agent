"""Shared file safety rules used by both tools and ACP shims.

Every guard here is defense-in-depth, NOT a security boundary: the terminal
tool runs as the same OS user and can read/write anything. The value is a
clear denial for models that respect tool errors plus a visible audit trail.
"""

from __future__ import annotations

import os
import re
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
# Round 3 settled the guard into TWO TIERS, because this tree is also forked
# for plain CLI / standalone-gateway deployments that never set an allowlist —
# there, a blanket ``/etc`` deny broke ordinary operator work for no security
# gain, while the credential files must stay unreadable regardless.
#
#   ① **Credential & privacy denials — ALWAYS on, and they outrank the
#      allowlist.** They do not depend on ``HERMES_READ_SAFE_ROOTS``:
#        * ``$HERMES_HOME`` / the Hermes root / ``~/.hermes``:
#          ``.env``, ``config.yaml``, ``auth.json``, ``state.db``,
#          ``channel_directory.json``, plus the ``memories/`` and
#          ``sessions/`` trees;
#        * any ``.env*`` / ``*.key`` / ``*.pem`` anywhere on disk;
#        * ``/proc/self/environ`` and ``/proc/<pid>/environ``.
#
#   ② **Directory-level denials — ONLY when ``HERMES_READ_SAFE_ROOTS`` is
#      set.** ``/etc``, ``/sys``, ``/dev``, the rest of ``/proc``, and the
#      remaining ``$HERMES_HOME`` subtrees (``logs/`` included). With no
#      allowlist configured these follow upstream behaviour and read
#      normally. An explicit allowlist root outranks this tier (but never ①).
#
#   ③ **Allowlist** — when ``HERMES_READ_SAFE_ROOTS`` is set, a read target
#      must resolve inside one of the listed roots. Anything else is refused
#      with a fixed, content-free payload that leaks neither the file's
#      existence nor its contents.
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

# ---------------------------------------------------------------------------
# Denial reason codes used by the readguard audit log (readguard.jsonl).
# The MODEL-facing payload stays uniform (``path_not_allowed``); these codes
# only ever reach the on-disk audit trail so an operator can tell a
# not-in-allowlist miss from a named-secret hit.
# ---------------------------------------------------------------------------
READGUARD_REASON_PATH = "path_not_allowed"
READGUARD_REASON_FILE = "denied_file"
READGUARD_REASON_EXPORT = "kb_export"
READGUARD_REASON_QUOTA = "kb_read_quota"

# --- Tier ①: credential & privacy denials — ALWAYS on -----------------------
# These win over the allowlist AND do not need one to be configured: a
# secret-bearing file stays unreadable even inside an explicitly allowed root
# (``/knowledge/.env``) and on a fork that never sets HERMES_READ_SAFE_ROOTS.

# Suffixes that always hold key material, anywhere on disk.
_READ_DENIED_SUFFIXES = (".key", ".pem")

# Files directly under $HERMES_HOME / the Hermes root / ~/.hermes. Deliberately
# NOT a global basename rule: a knowledge-base document that happens to be
# called ``config.yaml`` is ordinary content and must stay readable.
_HERMES_DENIED_FILES = (
    ".env",
    "config.yaml",
    "state.db",
    "auth.json",
    "channel_directory.json",
)

# Subtrees of $HERMES_HOME / the Hermes root / ~/.hermes that hold conversation
# history and long-term user memory — privacy, not just credentials, so they
# are denied whether or not an allowlist is configured.
_HERMES_ALWAYS_DENIED_SUBDIRS = ("memories", "sessions")

# Process environment blocks: every exported secret of a running process.
# ``/proc/self/environ`` and ``/proc/<pid>/environ`` for any pid.
_PROC_ENVIRON_RE = re.compile(r"^/proc/(?:self|\d+)/environ$")

# Exact absolute files that are never readable, independent of HERMES_HOME —
# the stock Haro container path, kept as a belt-and-braces literal.
_READ_DENIED_EXACT = ("/opt/data/config.yaml",)

# --- Tier ②: directory-level denials — only with an allowlist configured ----
# These LOSE to an explicit allowlist root (see
# :func:`classify_read_path_denial`): the Haro container sets
# ``HERMES_HOME=/opt/data`` and still hands the bot ``/opt/data/skills`` as a
# read root, so the whole-tree deny must not swallow the allowed subtree. And
# they are skipped entirely when no allowlist is set, so a CLI/standalone fork
# keeps upstream behaviour for ``/etc/hosts`` & co. Tier ① still applies in
# both cases.

# Absolute trees that are not readable through the managed tools.
_READ_DENIED_SYSTEM_PREFIXES = ("/proc", "/etc", "/sys", "/dev")

# Subdirectories of $HERMES_HOME / the Hermes root that are named explicitly.
# Already covered by the whole-tree prefix, but listed so the intent survives a
# future narrowing of that prefix (and so the tests can point at them).
# ``memories``/``sessions`` are NOT here — they were promoted to tier ①;
# ``logs`` (which holds readguard.jsonl) stays at this tier.
_HERMES_DENIED_SUBDIRS = ("logs",)


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


def _hermes_bases() -> list[str]:
    """$HERMES_HOME, the Hermes root and ``~/.hermes``, raw and realpath'd.

    The literal ``~/.hermes`` matters when ``HERMES_HOME`` has been pointed
    elsewhere but the stock profile still exists on disk.
    """
    bases: list[str] = []
    for base in (
        _hermes_home_path(),
        _hermes_root_path(),
        os.path.expanduser("~/.hermes"),
    ):
        for form in (str(base), os.path.realpath(base)):
            if form and form not in bases:
                bases.append(form)
    return bases


def _always_denied_prefixes() -> list[str]:
    """Tier ① trees: ``memories/`` and ``sessions/`` under every Hermes base."""
    prefixes: list[str] = []
    for base in _hermes_bases():
        for sub in _HERMES_ALWAYS_DENIED_SUBDIRS:
            try:
                candidate = os.path.join(base, sub)
            except (OSError, ValueError, TypeError):
                continue
            for form in (candidate, os.path.realpath(candidate)):
                if form and form not in prefixes:
                    prefixes.append(form)
    return prefixes


def _denied_prefixes() -> list[str]:
    """Tier ② directory trees, in both raw and symlink-resolved form.

    Only consulted when ``HERMES_READ_SAFE_ROOTS`` is configured.

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
    # ~/.hermes default.
    for base in _hermes_bases():
        if base not in prefixes:
            prefixes.append(base)
        # Named explicitly even though the whole-tree prefix above already
        # covers it: logs/ (which holds readguard.jsonl itself) must stay
        # denied if that prefix is ever narrowed.
        for sub in _HERMES_DENIED_SUBDIRS:
            try:
                candidate = os.path.join(base, sub)
            except (OSError, ValueError, TypeError):
                continue
            for form in (candidate, os.path.realpath(candidate)):
                if form and form not in prefixes:
                    prefixes.append(form)
    return prefixes


def _denied_exact_files() -> list[str]:
    """Explicitly named unreadable files (``$HERMES_HOME/config.yaml`` & co.)."""
    exact = list(_READ_DENIED_EXACT)
    for base in _hermes_bases():
        for name in _HERMES_DENIED_FILES:
            try:
                candidate = os.path.join(base, name)
            except (OSError, ValueError, TypeError):
                continue
            for form in (candidate, os.path.realpath(candidate)):
                if form and form not in exact:
                    exact.append(form)
    return exact


def _read_denied_file_hit(candidates: tuple[str, ...]) -> bool:
    """Tier ① — credential & privacy denial. Always on; outranks the allowlist.

    Fires for ``.env*`` / ``*.key`` / ``*.pem`` anywhere on disk, for the named
    ``$HERMES_HOME`` credential and state files, for the ``memories/`` and
    ``sessions/`` trees under any Hermes base, and for ``/proc/<pid>/environ``.

    Note what is deliberately absent: a bare ``config.yaml``/``state.db``
    basename elsewhere on disk. A knowledge-base article named ``config.yaml``
    is ordinary content, and blanket-denying the basename made it unreadable.
    """
    exact = _denied_exact_files()
    always = _always_denied_prefixes()
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

        if candidate in exact:
            return True

        # /proc/self/environ and /proc/<pid>/environ — the exported secrets of
        # any running process. Checked here (not at tier ②) so it stays denied
        # on a deployment that configures no allowlist at all.
        if _PROC_ENVIRON_RE.match(candidate):
            return True

        for prefix in always:
            try:
                if _under(candidate, prefix):
                    return True
            except (OSError, ValueError):
                continue
    return False


def _read_denied_prefix_hit(candidates: tuple[str, ...]) -> bool:
    """Tier ② directory-level denial — LOSES to an explicit allowlist root."""
    prefixes = _denied_prefixes()
    for candidate in candidates:
        if not candidate:
            continue
        for prefix in prefixes:
            try:
                if _under(candidate, prefix):
                    return True
            except (OSError, ValueError):
                continue
    return False


def _read_denylist_hit(candidates: tuple[str, ...]) -> bool:
    """Back-compat shim: True when either denial tier fires."""
    return _read_denied_file_hit(candidates) or _read_denied_prefix_hit(candidates)


def _allowlist_hit(resolved: str, roots: set[str]) -> bool:
    """True when ``resolved`` is one of ``roots`` or lives beneath one."""
    for root in roots:
        if resolved == root or resolved.startswith(root.rstrip(os.sep) + os.sep):
            return True
    return False


def classify_read_path_denial(path: str) -> Optional[str]:
    """Return the audit reason for refusing ``path``, or ``None`` when allowed.

    Precedence (round 3 — two tiers, because this tree is also forked for
    CLI / standalone-gateway deployments that configure no allowlist):

      1. ``HERMES_READ_SAFE_ROOTS_BYPASS`` — everything allowed.
      2. Unresolvable input — refused (fail closed).
      3. **Tier ① credential & privacy denials — always on, and they outrank
         the allowlist**: ``.env*`` / ``*.key`` / ``*.pem`` anywhere,
         ``/proc/<pid>/environ``, and under ``$HERMES_HOME`` / the Hermes root
         / ``~/.hermes`` the files ``config.yaml``, ``state.db``,
         ``auth.json``, ``channel_directory.json``, ``.env`` plus the
         ``memories/`` and ``sessions/`` trees.
      4. **No allowlist configured** — nothing further applies; upstream
         behaviour resumes (``/etc/hosts``, ``$HERMES_HOME/logs/x.log`` and
         ``~/.hermes/skills/x/SKILL.md`` all read normally).
      5. **Explicit allowlist roots** — a path under one is allowed even when
         it sits inside a tier ② denied tree (the Haro container sets
         ``HERMES_HOME=/opt/data`` *and* allowlists ``/opt/data/skills``).
      6. **Tier ② directory-level denials** (``/proc``, ``/etc``, ``/sys``,
         ``/dev``, ``$HERMES_HOME``, the Hermes root, ``~/.hermes`` — which
         covers ``logs/``).
      7. Anything else not under an allowlist root.
    """
    if is_read_safe_root_bypassed():
        return None

    resolved = _resolve_for_read_guard(path)
    if resolved is None:
        # Unresolvable input: fail closed.
        return READGUARD_REASON_PATH

    # The denylist is checked against BOTH the symlink-resolved path and the
    # merely-normalized one, so neither ``/etc/passwd`` (a symlink on macOS)
    # nor a symlink pointing INTO a denied tree can slip through.
    try:
        normalized = os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))
    except (OSError, ValueError):
        normalized = resolved
    candidates = (resolved, normalized)

    # Tier ①: always on, wins over everything below.
    if _read_denied_file_hit(candidates):
        return READGUARD_REASON_FILE

    roots = get_safe_read_roots()
    if not roots:
        # No allowlist configured (CLI / standalone fork): tier ② is off and
        # the guard stops here.
        return None

    if _allowlist_hit(resolved, roots):
        # Explicit allowlist beats the tier ② directory denies.
        return None

    # Tier ②. Subsumed by the closing "not under any root" refusal below, but
    # kept explicit so the denied trees stay legible and the two rules can
    # diverge later.
    if _read_denied_prefix_hit(candidates):
        return READGUARD_REASON_PATH

    return READGUARD_REASON_PATH


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

    The *reason* (allowlist miss vs. named-secret file) is deliberately NOT
    reflected here — use :func:`classify_read_path_denial` for the audit log.
    """
    if classify_read_path_denial(path) is None:
        return None
    return {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}


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

#: Interpreters whose ``-c`` / ``-e`` / ``-m`` operand is *code or a module
#: name*, never a file path. The operand is still swept for absolute paths
#: embedded inside it (``python -c 'open("/etc/passwd")'`` stays refused) —
#: only the "the whole operand is a path" reading is dropped.
_INTERPRETER_COMMANDS = frozenset({
    "python", "python3", "perl", "ruby", "node", "php", "sh", "bash", "zsh",
})

#: Flags after which an interpreter's operand is code / a module name.
_INTERPRETER_CODE_FLAGS = frozenset({"-c", "-e", "-m", "--command", "--eval"})

#: Directory basenames that hold executables. An ``argv[0]`` living directly
#: inside one is an *exec* path, not a path being read.
_EXEC_DIR_BASENAMES = frozenset({"bin", "sbin", "libexec"})


def is_exec_path_token(token: str) -> bool:
    """True when *token* names a program to run rather than a file to read.

    Only meaningful in ``argv[0]`` position. A bare command name (``python3``,
    ``cat``) qualifies, and so does an absolute path sitting directly inside a
    ``bin`` / ``sbin`` / ``libexec`` directory — ``/usr/bin/env``, ``/bin/sh``,
    ``/opt/hermes/.venv/bin/python``. Nothing else does: ``..`` traversals and
    any other absolute path invoked as a command (``/etc/passwd``) keep going
    through the ordinary read check, so the guard still fails closed on shapes
    it cannot vouch for.

    The exemption covers **exec only**. The very same path named as a read
    operand (``cat /opt/hermes/.venv/bin/python``, ``read_file``) is untouched
    by this function and stays refused.
    """
    if not token or not isinstance(token, str):
        return False
    if "://" in token:
        return False
    normalized = token.replace("\\", "/")
    if ".." in normalized.split("/"):
        return False
    if "/" not in normalized and not normalized.startswith("~"):
        # A bare name resolved through $PATH — not a path operand at all.
        return True
    if not normalized.startswith("/"):
        return False
    parent = os.path.basename(os.path.dirname(normalized))
    return parent in _EXEC_DIR_BASENAMES


def _strip_command_prefixes(argv: list[str]) -> list[str]:
    """Drop leading ``FOO=bar`` assignments and wrapper commands (``sudo``…)."""
    while argv and (
        _ENV_ASSIGN_TOKEN_RE.match(argv[0])
        or os.path.basename(argv[0]) in _COMMAND_PREFIXES
    ):
        argv = argv[1:]
    return argv


def _exec_path_tokens(command: str) -> set[str]:
    """The ``argv[0]`` tokens of *command*'s segments that are exec paths.

    Used to keep the whole-string baseline sweep from flagging an interpreter
    location (``/opt/hermes/.venv/bin/python -c …``) as a path being read. The
    per-segment operand pass is unaffected, so the same string appearing as an
    operand elsewhere in the command is still checked.
    """
    tokens: set[str] = set()
    for segment in _SEGMENT_SPLIT_RE.split(command or ""):
        segment = segment.strip()
        if not segment:
            continue
        try:
            argv = _shlex.split(segment, posix=True)
        except ValueError:
            argv = [tok.strip("'\"") for tok in segment.split()]
        # A wrapper spelled as a path (``/usr/bin/env FOO=1 prog``) is an exec
        # location too, so record it before it is stripped.
        stripped = _strip_command_prefixes(argv)
        for token in argv[: len(argv) - len(stripped)] + stripped[:1]:
            if is_exec_path_token(token):
                tokens.add(token)
    return tokens


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

    # Exec positions are not read positions: the interpreter/executable a
    # command runs (``/opt/hermes/.venv/bin/python -c …``) is exempted from the
    # baseline sweep below. Naming the same file as an *operand* still goes
    # through the per-command pass unchanged.
    exec_tokens = _exec_path_tokens(command)

    # Baseline pass over the raw string: any multi-segment absolute path
    # mentioned ANYWHERE in the command is checked, whatever the shell
    # structure around it. This is what catches shapes the tokenizer cannot
    # model — ``python -c 'open("/opt/data/config.yaml")'``, heredocs,
    # nested substitutions. Single-segment matches (``/etc`` on its own) are
    # left to the per-command pass so an incidental mention in a commit
    # message or a comment does not fail the whole command closed.
    for token in (command or "").split():
        stripped = token.strip("'\"")
        if stripped in exec_tokens:
            continue
        for found in _embedded_paths(stripped):
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
                if token in exec_tokens:
                    continue
                if _looks_absolute_or_escape(token):
                    candidates.append(token)
                candidates.extend(_embedded_paths(token))
            continue
        # Strip leading env assignments and wrapper commands.
        argv = _strip_command_prefixes(argv)
        if not argv:
            continue
        name = os.path.basename(argv[0])
        rest = argv[1:]
        is_read_cmd = name in _READ_COMMANDS
        is_interpreter = name in _INTERPRETER_COMMANDS
        skip_first_operand = (
            is_read_cmd
            and name in _SCRIPT_FIRST_COMMANDS
            and not any(flag in _PATTERN_FLAGS for flag in rest)
        )
        seen_operand = False
        previous = ""
        for token in rest:
            if not token or token == "-":
                previous = token
                continue
            # ``python -c '<code>'`` / ``python -m <module>``: the operand is
            # code, not a file. Absolute paths *inside* it are still swept
            # (see _embedded_paths below), only the whole-operand-is-a-path
            # reading is dropped.
            if is_interpreter and previous in _INTERPRETER_CODE_FLAGS:
                candidates.extend(_embedded_paths(token))
                previous = token
                seen_operand = True
                continue
            previous = token
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
    detail = get_command_read_denial_detail(command, cwd)
    if detail is None:
        return None
    return {"error": READ_PATH_DENIED_CODE, "message": READ_PATH_DENIED_MESSAGE}


def get_command_read_denial_detail(
    command: str, cwd: Optional[str] = None
) -> Optional[dict]:
    """Like :func:`get_command_read_denial` but names the offending operand.

    Returns ``{"error", "message", "path", "reason"}`` where ``path`` is the
    FIRST out-of-bounds path found in the command and ``reason`` is the audit
    code. Only the audit log consumes ``path``/``reason``; the model still gets
    the uniform two-key payload.
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
        reason = classify_read_path_denial(expanded)
        if reason is not None:
            return {
                "error": READ_PATH_DENIED_CODE,
                "message": READ_PATH_DENIED_MESSAGE,
                "path": os.path.normpath(expanded),
                "reason": reason,
            }
    return None


# ---------------------------------------------------------------------------
# Readguard audit log ($HERMES_HOME/logs/readguard.jsonl)
#
# Every refusal produced by the read guard, the knowledge-base export guard and
# the per-turn read quota appends ONE json line. The line records *what was
# refused and to whom* — never any file content, and never the command text.
#
# The log file itself lives under ``$HERMES_HOME/logs/``, which the
# directory-level denylist already refuses, so the agent cannot read back its
# own audit trail.
#
# Best effort by construction: a write failure (read-only volume, no space,
# permission) is swallowed. An audit trail that can break a security refusal is
# worse than a missing line.
# ---------------------------------------------------------------------------

READGUARD_LOG_DIRNAME = "logs"
READGUARD_LOG_BASENAME = "readguard.jsonl"


def get_readguard_log_path() -> str:
    """Absolute path of the readguard audit log under ``$HERMES_HOME``."""
    return os.path.join(
        str(_hermes_home_path()), READGUARD_LOG_DIRNAME, READGUARD_LOG_BASENAME
    )


def _session_field(name: str) -> str:
    """Read one ``HERMES_SESSION_*`` value without importing the gateway eagerly."""
    try:
        from gateway.session_context import get_session_env
        return (get_session_env(name, "") or "").strip()
    except Exception:  # noqa: BLE001 - audit must never break a refusal
        return (os.getenv(name, "") or "").strip()


def get_readguard_identity() -> dict:
    """Session / subject / platform for the audit line.

    Sourced from the gateway's per-task session context (``HERMES_SESSION_ID``,
    ``HERMES_SESSION_USER_ID``, ``HERMES_SESSION_PLATFORM`` with
    ``HERMES_SESSION_SOURCE`` as the CLI/TUI fallback). Anything missing is
    recorded as ``"unknown"`` rather than omitted, so every line has the same
    shape.
    """
    session = _session_field("HERMES_SESSION_ID") or _session_field("HERMES_SESSION_KEY")
    subject = (
        _session_field("HERMES_SESSION_USER_ID")
        or _session_field("HERMES_SESSION_USER_ID_ALT")
    )
    platform = _session_field("HERMES_SESSION_PLATFORM") or _session_field(
        "HERMES_SESSION_SOURCE"
    )
    return {
        "session": session or "unknown",
        "subject": subject or "unknown",
        "platform": platform or "unknown",
    }


def log_readguard_denial(tool: str, path: str, reason: str) -> None:
    """Append one audit line for a refusal. Never raises."""
    try:
        import datetime as _dt
        import json as _json

        identity = get_readguard_identity()
        record = {
            "ts": _dt.datetime.now(_dt.timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "tool": tool,
            "path": str(path or ""),
            "session": identity["session"],
            "subject": identity["subject"],
            "platform": identity["platform"],
            "reason": reason,
        }
        log_path = get_readguard_log_path()
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 - see module note: refusal wins over audit
        return


def normalize_audit_path(path: str) -> str:
    """Canonical form of ``path`` for the audit line; falls back to the input."""
    try:
        return os.path.normpath(os.path.abspath(os.path.expanduser(str(path))))
    except (OSError, ValueError, TypeError):
        return str(path)


# ---------------------------------------------------------------------------
# Knowledge-base roots (HERMES_KB_ROOTS)
#
# The bot is *supposed* to read knowledge files one at a time and answer from
# them. It is not supposed to hand a chat user the whole corpus. Two guards
# below share this notion of "the knowledge tree":
#
#   * the bulk-export guard (tar/zip/cp -r/rsync/... on a knowledge directory)
#   * the per-turn read quota
#
# Default: the ``HERMES_READ_SAFE_ROOTS`` entries whose basename is
# ``knowledge`` (Haro injects ``/knowledge`` there). Haro may also set
# ``HERMES_KB_ROOTS`` explicitly.
# ---------------------------------------------------------------------------

KB_ROOTS_ENV = "HERMES_KB_ROOTS"
KB_READ_PER_TURN_ENV = "HERMES_KB_READ_PER_TURN"
KB_READ_PER_TURN_DEFAULT = 20

KB_EXPORT_DENIED_CODE = "kb_export_forbidden"
KB_EXPORT_DENIED_MESSAGE = "知识库内容不提供整包导出"

KB_READ_QUOTA_CODE = "kb_read_quota"


def get_kb_roots() -> set[str]:
    """Return resolved knowledge-base roots."""
    env = os.getenv(KB_ROOTS_ENV, "")
    if env:
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
    # Fall back to the read roots that look like a knowledge tree.
    return {
        root for root in get_safe_read_roots()
        if os.path.basename(root.rstrip(os.sep)).lower() == "knowledge"
    }


def is_under_kb_root(path: str) -> bool:
    """True when ``path`` resolves inside a configured knowledge root."""
    roots = get_kb_roots()
    if not roots:
        return False
    resolved = _resolve_for_read_guard(path)
    if resolved is None:
        return False
    return _allowlist_hit(resolved, roots)


# ---------------------------------------------------------------------------
# Bulk-export guard
#
# Fail-closed by design: a command we cannot fully parse is refused as soon as
# it pairs an archiving/copying verb with a knowledge path that is the root
# itself, a directory, or a glob. Refusing a legitimate single-file copy that
# happens to look recursive is cheap; letting the corpus out is not.
# ---------------------------------------------------------------------------

#: Verbs that can move a whole tree somewhere else.
_EXPORT_ARCHIVE_COMMANDS = frozenset({
    "tar", "zip", "7z", "7za", "7zr", "rar", "gzip", "bzip2", "xz", "zstd",
})
_EXPORT_COPY_COMMANDS = frozenset({"cp", "install"})
#: Always bulk by nature — any knowledge operand is refused.
_EXPORT_SYNC_COMMANDS = frozenset({"rsync", "scp", "sftp"})

_EXPORT_RECURSIVE_FLAGS = frozenset({
    "-r", "-R", "-a", "-ar", "-ra", "-rp", "-Rp", "-av", "-avz", "-rf", "-Rf",
    "--recursive", "--archive",
})

#: Python one-liners that copy a tree or build an archive.
_EXPORT_PYTHON_MARKERS = (
    "copytree", "make_archive", "tarfile", "zipfile", "shutil.copy",
)

_GLOB_CHARS = ("*", "?", "[")


def _has_glob(token: str) -> bool:
    return any(ch in token for ch in _GLOB_CHARS)


def _glob_prefix(token: str) -> str:
    """Longest leading directory of a glob token that has no wildcard."""
    parts = token.split("/")
    kept: list[str] = []
    for part in parts:
        if _has_glob(part):
            break
        kept.append(part)
    return "/".join(kept) or "/"


def _abs_for_export(token: str, base: str) -> Optional[str]:
    try:
        expanded = os.path.expanduser(token)
    except (OSError, ValueError, TypeError):
        return None
    if not os.path.isabs(expanded):
        try:
            expanded = os.path.join(base, expanded)
        except (OSError, ValueError, TypeError):
            return None
    return _resolve_for_read_guard(expanded)


def _kb_operand(token: str, base: str, roots: set[str]) -> Optional[str]:
    """Return the resolved path when ``token`` names something in a KB root."""
    if not token or token.startswith("-") or "://" in token:
        return None
    probe = _glob_prefix(token) if _has_glob(token) else token
    resolved = _abs_for_export(probe, base)
    if resolved is None:
        return None
    if _allowlist_hit(resolved, roots):
        return resolved
    return None


def _is_bulk_source(token: str, resolved: str, roots: set[str]) -> bool:
    """True when the operand designates a whole tree rather than one file."""
    if _has_glob(token):
        return True
    if resolved in roots:
        return True
    if token.endswith("/"):
        return True
    try:
        if os.path.isdir(resolved):
            return True
    except OSError:
        pass
    # Fail closed: a path we cannot stat and that carries no file extension is
    # treated as a directory.
    if not os.path.exists(resolved) and not os.path.splitext(resolved)[1]:
        return True
    return False


def get_command_export_denial(
    command: str, cwd: Optional[str] = None
) -> Optional[dict]:
    """Refuse a command that would export the knowledge base in bulk.

    Returns ``{"error": "kb_export_forbidden", "message": ...}`` (plus a private
    ``path`` used only for the audit line) or ``None``.

    Recognised shapes: ``tar``/``zip``/``7z`` over a knowledge directory,
    ``cp -r|-R|-a``, ``rsync``/``scp``, ``find <kb> ... -exec cp``,
    ``find <kb> | xargs cp``, ``python -c '...shutil.copytree/make_archive...'``,
    and a ``cat <kb>/*`` redirected to a file. A single-file
    ``cp /knowledge/x/a.md /out/`` is NOT refused — the per-turn read quota is
    what bounds that path.
    """
    if not command or not isinstance(command, str):
        return None
    roots = get_kb_roots()
    if not roots:
        return None

    base = cwd or os.getcwd()

    def deny(path: str) -> dict:
        return {
            "error": KB_EXPORT_DENIED_CODE,
            "message": KB_EXPORT_DENIED_MESSAGE,
            "path": path,
            "reason": READGUARD_REASON_EXPORT,
        }

    # ``find <kb> ... -exec cp`` / ``find <kb> ... | xargs cp`` and Python
    # one-liners are easier to spot on the whole string than per segment.
    lowered = command.lower()
    whole_tokens: list[str] = []
    for token in command.split():
        whole_tokens.append(token.strip("'\"();"))

    def first_kb_token(bulk_only: bool = False) -> Optional[str]:
        for token in whole_tokens:
            for candidate in [token] + _embedded_paths(token):
                resolved = _kb_operand(candidate, base, roots)
                if resolved is None:
                    continue
                if bulk_only and not _is_bulk_source(candidate, resolved, roots):
                    continue
                return resolved
        return None

    if "-exec" in whole_tokens or "xargs" in whole_tokens:
        if any(v in lowered for v in ("cp ", "cp\t", "rsync", "tar ", "install ")):
            hit = first_kb_token()
            if hit:
                return deny(hit)

    if any(marker in command for marker in _EXPORT_PYTHON_MARKERS):
        hit = first_kb_token()
        if hit:
            return deny(hit)

    for segment in _SEGMENT_SPLIT_RE.split(command):
        segment = segment.strip()
        if not segment:
            continue
        try:
            argv = _shlex.split(segment, posix=True)
        except ValueError:
            argv = [t.strip("'\"") for t in segment.split()]
        while argv and (
            _ENV_ASSIGN_TOKEN_RE.match(argv[0])
            or os.path.basename(argv[0]) in _COMMAND_PREFIXES
        ):
            argv = argv[1:]
        if not argv:
            continue
        name = os.path.basename(argv[0])
        rest = argv[1:]

        if name in _EXPORT_SYNC_COMMANDS:
            for token in rest:
                resolved = _kb_operand(token, base, roots)
                if resolved is not None:
                    return deny(resolved)
            continue

        if name in _EXPORT_ARCHIVE_COMMANDS:
            for token in rest:
                resolved = _kb_operand(token, base, roots)
                if resolved is None:
                    continue
                if _is_bulk_source(token, resolved, roots):
                    return deny(resolved)
            continue

        if name in _EXPORT_COPY_COMMANDS:
            recursive = any(
                flag in _EXPORT_RECURSIVE_FLAGS
                or (
                    flag.startswith("-")
                    and not flag.startswith("--")
                    and any(c in flag[1:] for c in "rRa")
                )
                for flag in rest
                if flag.startswith("-")
            )
            for token in rest:
                resolved = _kb_operand(token, base, roots)
                if resolved is None:
                    continue
                if recursive or _is_bulk_source(token, resolved, roots):
                    return deny(resolved)
            continue

        # ``cat /knowledge/x/* > /out/all.md`` — a glob read funnelled into a
        # file is an export in everything but name.
        if name in ("cat", "tail", "head") and ">" in segment:
            for token in rest:
                if not _has_glob(token):
                    continue
                resolved = _kb_operand(token, base, roots)
                if resolved is not None:
                    return deny(resolved)

    return None


# ---------------------------------------------------------------------------
# Per-turn knowledge read quota (HERMES_KB_READ_PER_TURN)
#
# A chat user cannot ask for the corpus in one archive (guard above), so the
# next-best exfiltration is "read me every file". The quota bounds how many
# knowledge files ONE turn may open through ``read_file``; ``search_files`` is
# deliberately not counted, since pointing the model at the retrieval tool is
# exactly the behaviour the refusal asks for.
#
# Turn boundary: ``agent/turn_context.py`` calls :func:`reset_kb_read_quota`
# once per turn, right where the turn id is minted and ``note_turn_start``
# fires — i.e. once per inbound user message. Counters are process-local and
# keyed by session id, so concurrent sessions do not share a budget.
# ---------------------------------------------------------------------------

import threading as _threading

_KB_READ_COUNTS: dict = {}
_KB_READ_LOCK = _threading.Lock()


def get_kb_read_limit() -> int:
    """Per-turn knowledge read budget (``HERMES_KB_READ_PER_TURN``)."""
    raw = (os.getenv(KB_READ_PER_TURN_ENV, "") or "").strip()
    if not raw:
        return KB_READ_PER_TURN_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return KB_READ_PER_TURN_DEFAULT
    return value if value >= 0 else KB_READ_PER_TURN_DEFAULT


def _kb_quota_session_key() -> str:
    return get_readguard_identity()["session"]


def reset_kb_read_quota(session_id: Optional[str] = None) -> None:
    """Clear the per-turn counter at a turn boundary. Never raises.

    Called with the agent's ``session_id``; the counting side derives its key
    from the gateway session context, which can spell the same session
    differently (or not at all, under the CLI), so both keys are cleared.
    """
    try:
        keys = {_kb_quota_session_key()}
        if session_id:
            keys.add(str(session_id))
        with _KB_READ_LOCK:
            for key in keys:
                _KB_READ_COUNTS.pop(key, None)
    except Exception:  # noqa: BLE001
        return


def reset_all_kb_read_quotas() -> None:
    """Drop every counter (test helper / process-wide reset)."""
    with _KB_READ_LOCK:
        _KB_READ_COUNTS.clear()


def get_kb_read_quota_denial() -> dict:
    limit = get_kb_read_limit()
    return {
        "error": KB_READ_QUOTA_CODE,
        "message": (
            f"本轮读取知识库文件已达上限（{limit}），"
            "请改用检索工具定位后再读"
        ),
    }


def check_kb_read_quota(path: str) -> Optional[dict]:
    """Count one ``read_file`` against the per-turn knowledge budget.

    Returns the refusal payload once the budget is spent, else ``None`` (and
    charges the read). Non-knowledge paths are never counted, and
    ``HERMES_READ_SAFE_ROOTS_BYPASS`` lifts the limit entirely.
    """
    try:
        if is_read_safe_root_bypassed():
            return None
        if not is_under_kb_root(path):
            return None
        limit = get_kb_read_limit()
        if limit <= 0:
            return None
        key = _kb_quota_session_key()
        with _KB_READ_LOCK:
            used = _KB_READ_COUNTS.get(key, 0)
            if used >= limit:
                return get_kb_read_quota_denial()
            _KB_READ_COUNTS[key] = used + 1
        return None
    except Exception:  # noqa: BLE001 - a broken counter must not block reads
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
