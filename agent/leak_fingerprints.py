"""Outbound fingerprint guard: never let the system prompt out verbatim.

The reasoning-leak rules in :mod:`agent.leak_guard` catch the *shape* of a
thought spilling into ``content``.  They cannot catch the other observed exit
(2026-09-09, two sessions): a user talks the agent into ``read_file`` on its own
``config.yaml`` and the agent obligingly quotes it back, line by line, as a
perfectly well-formed answer.  No rule about how a reply *opens* will ever see
that.

So the operator gets a second, content-addressed gate.  Haro pushes a digest of
whatever must never be echoed to ``$HERMES_HOME/guard/prompt-fingerprints.json``
(via WriteHome, tmp+rename).  Only hashes travel, never the protected text — the
file is safe to ship to every gateway.  Absent or unreadable file ⇒ the hook is
inert (fail-open: a broken digest must never mute a gateway); the file's mtime
is polled so an updated digest takes effect without a restart.

THE FORMAT IS A CROSS-LANGUAGE CONTRACT with the Haro (Go) generator.  Both
sides must produce byte-identical hashes, so every step is pinned:

* normalization — ``\\r\\n`` → ``\\n``; ``unicodedata.normalize("NFKC")``;
  per line: trim, collapse runs of whitespace to a single space, lowercase
  **A-Z only** (never ``str.lower()``, which would also fold non-ASCII scripts
  the Go side leaves alone); split on ``\\n``.
* line fingerprint — a normalized line of **≥ 12 runes** is registered as the
  first 16 bytes of ``sha256(utf8(line))``, i.e. 32 lowercase hex chars.
* 8-gram fingerprint — a rune window of width 8 over a normalized line, hashed
  with ``fnv1a-64`` over the window's UTF-8 bytes (offset basis
  0xcbf29ce484222325, prime 0x100000001b3, truncated to 64 bits), rendered as
  16 lowercase hex chars.  Windows never straddle a line break, so a line of
  fewer than 8 runes contributes no gram.  The file-wide list is deduplicated
  and sorted ascending.

FORMAT v2 (contract §6, 2026-09-11) adds three BUILD-side gates, applied by the
Go generator and by this module alike, so both sides protect the same set:
fenced code blocks contribute nothing; a line that is bare path/identifier runes
after its markdown markers are stripped, or that holds no non-ASCII rune and is
under 24 runes, contributes nothing; and only a window holding a non-ASCII rune
becomes a gram.  ``build_fingerprints(..., version=1)`` still writes the
original unfiltered digest, and both versions are read.

On-disk shape::

    {"version": 2, "botId": "…", "generatedAt": "<RFC3339>",
     "normalize": "nfkc+trim+collapse-ws+ascii-lower+lowentropy-v2",
     "line":  {"hash": "sha256-16", "minRunes": 12},
     "ngram": {"n": 8, "unit": "rune", "hash": "fnv1a-64"},
     "sources": [{"id": "answer_rules", "lines": 12, "ngrams": 430}, …],
     "lines":  ["<32 hex>", …],
     "ngrams": ["<16 hex>", …]}

Matching is deliberately asymmetric: **one** protected line is enough (a whole
line reproduced verbatim is not a coincidence), while 8-grams need **three
hits that are adjacent** in the reply — adjacency being consecutive window
positions or a gap of at most one.  A single 8-gram of ordinary Chinese or a
common English phrase collides with innocent text all the time; three of them
in a row do not.

``scripts/replyguard_fingerprints.py`` builds the file and is the reference
implementation for the writer side.

LOW-ENTROPY WINDOWS ARE NOT EVIDENCE (2026-09-11).  Haro's production digest
fingerprints a bot's whole bound skill, so ``/knowled`` / ``kb_searc`` and the
rest of the SKILL.md's paths and tool names ended up protected — while the same
bot's answer rules oblige every knowledge answer to cite the ``/knowledge/…``
path it read.  Three of four ordinary questions were redacted in the regression.
The matching side therefore discards a window that is all
:data:`IDENTIFIER_CHARS`, or pure ASCII with no space, or built from ≤ 3
distinct runes, before the VERDICT, and skips a line that is nothing but a path;
an adjacent run convicts only if one of its surviving windows holds a non-ASCII
rune or two space-separated words.  A run that fails that bar is real but
harmless and is reported as :attr:`FingerprintScanner.ascii_run`, which the
reply guard writes as one ``reply.audited`` line while the reply goes out.  The
generator functions are untouched: they are the byte-for-byte contract with the
Go side, and a window that is never looked up can never match anyway.

SECOND SOURCE: the container's own prompt.  Haro's digest only covers what Haro
pushed — answer rules, identity, status phrases, bound skills.  The SOUL block,
the role rules and the tool briefs are baked into the hermes image and never
travel through Haro at all, which is precisely what a maintainer bot has to
lose.  So :func:`register_system_prompt` runs the same algorithm over the system
prompt the container just assembled, keeps the result in RAM (never on disk,
neither the text nor the hashes), and :func:`combined_store` matches against the
UNION of the two.  It is cached by the prompt's sha256, so an identity patch or
a post-compression rebuild re-arms the gate on the new bytes and disarms it on
the old.  ``HERMES_GUARD_SELF_FINGERPRINT=0`` turns the half off; with no Haro
file at all, it is the only protection there is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import unicodedata
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

# ── contract constants (mirrored in the Haro Go generator) ─────────────
#: Digest format written by default: v2 applies the §6 low-entropy build-side
#: filter below.  v1 (no filter) is still written on request and still read.
FORMAT_VERSION = 2
FORMAT_VERSION_V1 = 1
SUPPORTED_FORMAT_VERSIONS = frozenset({1, 2})
NORMALIZE_ID_V1 = "nfkc+trim+collapse-ws+ascii-lower"
NORMALIZE_ID_V2 = "nfkc+trim+collapse-ws+ascii-lower+lowentropy-v2"
#: Backwards-compatible alias: the v1 normalization id.
NORMALIZE_ID = NORMALIZE_ID_V1
LINE_HASH_ID = "sha256-16"
LINE_HASH_BYTES = 16
#: A normalized line shorter than this is not registered: "是"/"步骤如下" recur
#: in innocent replies and would fire the line rule on everything.
MIN_LINE_RUNES = 12
NGRAM_N = 8
NGRAM_UNIT = "rune"
NGRAM_HASH_ID = "fnv1a-64"

FNV64_OFFSET_BASIS = 0xCBF29CE484222325
FNV64_PRIME = 0x100000001B3
FNV64_MASK = 0xFFFFFFFFFFFFFFFF

#: Adjacent 8-gram hits needed before a reply counts as reproducing the source.
NGRAM_RUN_THRESHOLD = 3
#: Two matched window positions are "adjacent" when their distance is ≤ this
#: (1 = consecutive, 2 = exactly one window skipped).
NGRAM_ADJACENT_GAP = 2
#: One whole protected line is enough.
LINE_HIT_THRESHOLD = 1

# ── low-entropy windows (2026-09-11 false-positive fix) ────────────────
# Haro's production digest fingerprints the WHOLE bound skill (4470 of a bot's
# 4555 grams came from one SKILL.md), so ``/knowled``, ``knowledg``, ``kb_searc``
# … all landed in the protected set.  The same bot's answer rules then REQUIRE
# every knowledge answer to end with the ``/knowledge/…`` path it used, and the
# reply guard ate three of four ordinary questions in the 2026-09-11 regression.
# A path segment, a tool name or a bare identifier is not evidence of anything:
# it recurs in innocent replies by construction.  So they never convict.
#
# The filter is applied on the MATCHING side and on the container's own
# self-fingerprint side.  It is deliberately NOT applied in
# :func:`ngram_fingerprints` / :func:`line_fingerprints`, which are the
# cross-language contract with the Haro Go generator (and are pinned by
# ``tests/agent/fixtures/guard_vectors.json``): Haro's ngrams for a given source
# must stay byte-identical.  Filtering the reply's windows before the lookup is
# equivalent — a window that is never looked up can never match.

#: Runes that on their own spell a path / URL / code identifier.
IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_/.:-"
)
#: A window built from this few distinct runes carries no evidence ("------").
MAX_LOW_ENTROPY_DISTINCT = 3


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in text)


def _word_count(text: str) -> int:
    """Space-separated words in ``text`` (the normalization already collapsed runs)."""
    return len([part for part in text.split(" ") if part])


def is_low_entropy_window(window: str) -> bool:
    """True when an 8-rune window is too common to be evidence of a leak.

    Any one of:

    * ≤ :data:`MAX_LOW_ENTROPY_DISTINCT` distinct runes (``--------``);
    * every rune in :data:`IDENTIFIER_CHARS` (``/knowled``, ``kb_searc``);
    * pure ASCII with no space at all (an identifier in any other alphabet soup).
    """
    if not window:
        return True
    if len(set(window)) <= MAX_LOW_ENTROPY_DISTINCT:
        return True
    if all(ch in IDENTIFIER_CHARS for ch in window):
        return True
    return _is_ascii(window) and " " not in window


def window_is_convicting(window: str) -> bool:
    """True when a matched window may carry a run to a conviction.

    A window earns that only by holding a non-ASCII rune (Chinese prose — the
    thing the digest actually protects) or two space-separated words.  A run of
    bare ASCII fragments is audited, never convicted.
    """
    return (not _is_ascii(window)) or _word_count(window) >= 2


def is_low_entropy_line(normalized_line: str) -> bool:
    """True when a whole normalized line is just a path / URL / identifier.

    ``/knowledge/canway-it-support/guides/access/vpn-user-guide.md`` on a line of
    its own is a citation, not the system prompt leaking.
    """
    if not normalized_line:
        return True
    if " " in normalized_line:
        return False
    return all(ch in IDENTIFIER_CHARS for ch in normalized_line)


def convicting_matches(matches: dict) -> dict:
    """The subset of ``{position: window}`` that may carry a conviction.

    Low-entropy windows are looked up like any other — the run they form is
    worth an audit line — but they are removed before the verdict, so a cited
    path can never redact a reply on its own.
    """
    return {
        position: window
        for position, window in matches.items()
        if not is_low_entropy_window(window)
    }

# ── §6 low-entropy build-side filter, v2 (2026-09-11) ──────────────────
# The match-side filter above keeps a *cited path* from redacting a reply, but
# the low-entropy material is still in the digest, and both sides have to agree
# on what is in it.  Contract §6 therefore moves three gates to the BUILD side,
# where Haro's Go generator applies exactly the same three:
#
#   1. a fenced code block (``` / ~~~ up to its matching fence, fences included,
#      an unterminated fence running to EOF) contributes nothing at all;
#   2. a line whose markdown markers have been stripped is dropped when it is
#      nothing but path/URL/identifier runes with no space (``ascii_ident``), or
#      when it holds no non-ASCII rune at all and is shorter than 24 runes
#      (``short_ascii``);
#   3. an 8-rune window with no non-ASCII rune contributes no gram.
#
# Two things are deliberately NOT moved.  The marker stripping decides only
# whether a line is kept — the hashes are still taken over the whole normalized
# line, so a line that survives both versions keeps the same fingerprint.  And
# the match-side filter stays in place as a double safety, because a v1 digest
# (Haro not yet upgraded) still has to be survivable.

#: Markdown line markers stripped from both ends before the v2 line tests.
MD_MARKER_CHARS = "-*>#|"
#: An ordered-list prefix (``1.`` / ``2)``) at the head of a line.
_ORDERED_LIST_RE = re.compile(r"\d+[.)](?=\s|$)")
#: Runes that on their own spell a path / URL / query string / identifier.
#: A superset of :data:`IDENTIFIER_CHARS` — §6 also covers ``?a=b&c#d``.
V2_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_/.:-+=?&%#@~"
)
#: An all-ASCII line shorter than this carries no evidence ("Use kb_search first.").
V2_MIN_ASCII_LINE_RUNES = 24
#: Opening/closing code fences.
FENCE_MARKERS = ("```", "~~~")

#: Drop reasons recorded by :func:`classify_lines_v2` (and by the v2 vectors).
DROP_FENCED = "fenced"
DROP_ASCII_IDENT = "ascii_ident"
DROP_SHORT_ASCII = "short_ascii"


def strip_markdown_markers(normalized_line: str) -> str:
    """Strip leading/trailing markdown markers (``- * > # |``, ``1.``/``2)``).

    Only the v2 keep/drop decision looks at the result; the fingerprints are
    still taken over the unstripped normalized line.
    """
    text = normalized_line.strip()
    while text:
        before = text
        match = _ORDERED_LIST_RE.match(text)
        if match:
            text = text[match.end():].lstrip(" ")
        text = text.lstrip(MD_MARKER_CHARS + " ")
        if text == before:
            break
    return text.rstrip(MD_MARKER_CHARS + " ").strip()


def fence_marker(normalized_line: str) -> Optional[str]:
    """The fence marker a line opens/closes with, or ``None``."""
    for marker in FENCE_MARKERS:
        if normalized_line.startswith(marker):
            return marker
    return None


def line_drop_reason_v2(normalized_line: str) -> Optional[str]:
    """Why §6 drops this (non-fenced) normalized line, or ``None`` to keep it."""
    text = strip_markdown_markers(normalized_line)
    if not text:
        return DROP_SHORT_ASCII
    if " " not in text and all(ch in V2_IDENTIFIER_CHARS for ch in text):
        return DROP_ASCII_IDENT
    if _is_ascii(text) and len(text) < V2_MIN_ASCII_LINE_RUNES:
        return DROP_SHORT_ASCII
    return None


def window_is_registered_v2(window: str) -> bool:
    """§6 rule 3 — only a window holding a non-ASCII rune becomes a gram."""
    return not _is_ascii(window)


def classify_lines_v2(text: str) -> list[tuple[str, Optional[str]]]:
    """``[(normalized_line, drop_reason or None), …]``, empty lines omitted.

    The single place the three §6 gates are decided, so the generator, the
    self digest and the vectors can never disagree with one another.
    """
    out: list[tuple[str, Optional[str]]] = []
    open_fence: Optional[str] = None
    for line in normalize_lines(text):
        if open_fence is not None:
            if line:
                out.append((line, DROP_FENCED))
            if line.startswith(open_fence):
                open_fence = None
            continue
        marker = fence_marker(line)
        if marker is not None:
            open_fence = marker
            out.append((line, DROP_FENCED))
            continue
        if not line:
            continue
        out.append((line, line_drop_reason_v2(line)))
    return out


def kept_lines_v2(text: str) -> list[str]:
    """The normalized lines §6 keeps, in order."""
    return [line for line, reason in classify_lines_v2(text) if reason is None]


#: Env switch for the container-side self fingerprints (default on; "0"/"false"/
#: "no"/"off" turns them off and leaves only whatever Haro pushed).
SELF_FINGERPRINT_ENV = "HERMES_GUARD_SELF_FINGERPRINT"
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})

#: Which digest a conviction came from (optional ``source`` field of the report).
SOURCE_HARO = "haro"
SOURCE_SELF = "self"
SOURCE_BOTH = "both"

#: Re-normalizing an unterminated line on every delta is O(len); above this the
#: provisional (pre-newline) check is skipped and the line is judged on flush.
MAX_PENDING_LINE_CHARS = 4000

GUARD_DIRNAME = "guard"
FINGERPRINTS_BASENAME = "prompt-fingerprints.json"

_WHITESPACE_RE = re.compile(r"\s+")


# ── normalization ──────────────────────────────────────────────────────
def ascii_lower(text: str) -> str:
    """Lowercase ``A-Z`` and nothing else.

    ``str.lower()`` also folds Greek, Cyrillic, Deseret … and would silently
    diverge from the Go side, which lowercases the ASCII range only.
    """
    return "".join(
        chr(ord(ch) + 32) if "A" <= ch <= "Z" else ch for ch in text
    )


def normalize_line(line: str) -> str:
    """trim → collapse whitespace runs → ASCII-lowercase (NFKC already applied)."""
    if not isinstance(line, str):
        return ""
    return ascii_lower(_WHITESPACE_RE.sub(" ", line.strip()))


def normalize_lines(text: str) -> list[str]:
    """The contract normalization, as the list of normalized lines."""
    if not isinstance(text, str) or not text:
        return []
    unified = unicodedata.normalize("NFKC", text.replace("\r\n", "\n"))
    return [normalize_line(line) for line in unified.split("\n")]


def normalize_text(text: str) -> str:
    """The contract normalization, as one ``\\n``-joined string."""
    return "\n".join(normalize_lines(text))


# ── hashes ─────────────────────────────────────────────────────────────
def line_hash(normalized_line: str) -> str:
    """First 16 bytes of ``sha256(utf8(line))`` as 32 lowercase hex chars."""
    return hashlib.sha256(normalized_line.encode("utf-8")).digest()[
        :LINE_HASH_BYTES
    ].hex()


def fnv1a64(data: bytes) -> int:
    """FNV-1a, 64-bit, over raw bytes."""
    digest = FNV64_OFFSET_BASIS
    prime, mask = FNV64_PRIME, FNV64_MASK
    for byte in data:
        digest = ((digest ^ byte) * prime) & mask
    return digest


def ngram_hash(window: str) -> str:
    """``fnv1a-64`` of a window's UTF-8 bytes, as 16 lowercase hex chars."""
    return f"{fnv1a64(window.encode('utf-8')):016x}"


def line_windows(normalized_line: str) -> list[str]:
    """Every rune window of width :data:`NGRAM_N` inside one normalized line."""
    if len(normalized_line) < NGRAM_N:
        return []
    return [
        normalized_line[i : i + NGRAM_N]
        for i in range(len(normalized_line) - NGRAM_N + 1)
    ]


def _check_version(version: int) -> int:
    version = int(version)
    if version not in SUPPORTED_FORMAT_VERSIONS:
        raise ValueError(f"unsupported fingerprint format version: {version!r}")
    return version


def build_lines(text: str, version: int = FORMAT_VERSION) -> list[str]:
    """The normalized lines a digest of ``version`` is built from."""
    if _check_version(version) == FORMAT_VERSION_V1:
        return [line for line in normalize_lines(text) if line]
    return kept_lines_v2(text)


def line_fingerprints(text: str, version: int = FORMAT_VERSION) -> list[str]:
    """Registered line hashes of ``text`` (deduplicated, ascending).

    ``version=2`` (the default) applies the §6 build-side filter first;
    ``version=1`` reproduces the original unfiltered contract byte for byte.
    """
    return sorted(
        {
            line_hash(line)
            for line in build_lines(text, version)
            if len(line) >= MIN_LINE_RUNES
        }
    )


def ngram_fingerprints(text: str, version: int = FORMAT_VERSION) -> list[str]:
    """8-gram hashes of ``text`` (deduplicated, ascending).

    ``version=2`` (the default) drops the lines §6 drops and then keeps only the
    windows holding a non-ASCII rune; ``version=1`` keeps every window.
    """
    # Windows are deduplicated BEFORE hashing: a 10 KB prompt repeats plenty of
    # them, and fnv1a over a window is the whole cost of generation.
    keep = (
        (lambda _window: True)
        if _check_version(version) == FORMAT_VERSION_V1
        else window_is_registered_v2
    )
    windows: set[str] = set()
    for line in build_lines(text, version):
        windows.update(window for window in line_windows(line) if keep(window))
    return sorted(ngram_hash(window) for window in windows)


def line_fingerprints_filtered(text: str) -> list[str]:
    """The v2 line hashes — the set the container's own digest registers.

    Kept as a name because the self digest used to filter on its own; it is now
    exactly :func:`line_fingerprints` at v2, so the two sides cannot drift.
    """
    return line_fingerprints(text, FORMAT_VERSION)


def ngram_fingerprints_filtered(text: str) -> list[str]:
    """The v2 8-gram hashes — see :func:`line_fingerprints_filtered`."""
    return ngram_fingerprints(text, FORMAT_VERSION)


def _rfc3339_now() -> str:
    import datetime as _dt

    return (
        _dt.datetime.now(_dt.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def build_fingerprints(
    sources: Sequence[tuple[str, str]],
    *,
    bot_id: str,
    generated_at: Optional[str] = None,
    version: int = FORMAT_VERSION,
) -> dict:
    """The on-disk digest for ``[(source_id, text), …]``. Hashes only, never text.

    ``version=2`` (the default) is the §6 low-entropy build; pass
    ``version=1`` for the original unfiltered contract.
    """
    version = _check_version(version)
    lines: set[str] = set()
    ngrams: set[str] = set()
    manifest = []
    for source_id, text in sources:
        source_lines = line_fingerprints(text, version)
        source_ngrams = ngram_fingerprints(text, version)
        lines.update(source_lines)
        ngrams.update(source_ngrams)
        manifest.append(
            {
                "id": str(source_id),
                "lines": len(source_lines),
                "ngrams": len(source_ngrams),
            }
        )
    return {
        "version": version,
        "botId": str(bot_id or ""),
        "generatedAt": generated_at or _rfc3339_now(),
        "normalize": (
            NORMALIZE_ID_V1 if version == FORMAT_VERSION_V1 else NORMALIZE_ID_V2
        ),
        "line": {"hash": LINE_HASH_ID, "minRunes": MIN_LINE_RUNES},
        "ngram": {"n": NGRAM_N, "unit": NGRAM_UNIT, "hash": NGRAM_HASH_ID},
        "sources": manifest,
        "lines": sorted(lines),
        "ngrams": sorted(ngrams),
    }


def fingerprints_path() -> str:
    """``$HERMES_HOME/guard/prompt-fingerprints.json``."""
    home = os.environ.get("HERMES_HOME")
    if not home:
        try:
            import hermes_constants

            home = str(hermes_constants.get_hermes_home())
        except Exception:
            home = os.path.expanduser("~/.hermes")
    return os.path.join(home, GUARD_DIRNAME, FINGERPRINTS_BASENAME)


class FingerprintStore:
    """The loaded digest, reloaded whenever the file's mtime/size changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stamp: Optional[tuple] = None
        self._path: Optional[str] = None
        self.bot_id: str = ""
        self.format_version: int = 0
        self.ngrams: frozenset[str] = frozenset()
        self.lines: frozenset[str] = frozenset()

    @property
    def empty(self) -> bool:
        return not (self.ngrams or self.lines)

    def classify(self, line_hits: Iterable[str], ngram_hits: Iterable[str]) -> str:
        """Which digest the hits came from — always Haro's, for this store."""
        return SOURCE_HARO

    def _clear(self) -> None:
        self.ngrams = self.lines = frozenset()
        self.bot_id = ""
        self.format_version = 0

    def refresh(self, path: Optional[str] = None) -> "FingerprintStore":
        """Re-read the digest if it changed. Never raises: a broken file is
        logged and treated as absent, because failing closed here would mute
        every reply on the gateway."""
        path = path or fingerprints_path()
        with self._lock:
            try:
                stat = os.stat(path)
                stamp = (path, stat.st_mtime_ns, stat.st_size)
            except OSError:
                if self._stamp is not None or self._path != path:
                    self._stamp, self._path = None, path
                    self._clear()
                return self
            if stamp == self._stamp:
                return self
            self._stamp, self._path = stamp, path
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, dict):
                    raise ValueError("fingerprint digest is not an object")
                self.bot_id = str(data.get("botId") or "")
                try:
                    self.format_version = int(data.get("version") or 0)
                except (TypeError, ValueError):
                    self.format_version = 0
                # v1 and v2 differ only in WHAT was fingerprinted; the hash
                # encoding is identical, so both read the same way (and an
                # unknown future version still loads rather than going inert —
                # the match side's own low-entropy filter is the backstop).
                if self.format_version not in SUPPORTED_FORMAT_VERSIONS:
                    logger.warning(
                        "Reply guard: fingerprint digest %s declares version %r "
                        "(known: %s) — reading its hashes anyway.",
                        path, self.format_version,
                        sorted(SUPPORTED_FORMAT_VERSIONS),
                    )
                self.lines = frozenset(str(h) for h in (data.get("lines") or []))
                self.ngrams = frozenset(str(h) for h in (data.get("ngrams") or []))
                logger.info(
                    "Reply guard: loaded %d line and %d 8-gram fingerprints "
                    "(bot=%s, format v%d) from %s",
                    len(self.lines), len(self.ngrams), self.bot_id or "-",
                    self.format_version, path,
                )
            except Exception:
                logger.warning(
                    "Reply guard: unreadable fingerprint file %s — hook stays inert.",
                    path, exc_info=True,
                )
                self._clear()
            return self


_STORE = FingerprintStore()


def store(path: Optional[str] = None) -> FingerprintStore:
    """The process-wide store, refreshed against the file on disk."""
    return _STORE.refresh(path)


# ── container-side self fingerprints ───────────────────────────────────
# Haro's digest covers what Haro knows it pushed (answer rules, identity, status
# phrases, bound skills).  The bulk of what the model is actually holding — the
# SOUL block, the role rules, the tool briefs, every operator patch baked into
# the image — never travels through Haro at all, so no file can fingerprint it.
# The container can: by the time a turn is sent, the assembled system prompt IS
# in memory, and running the very same contract algorithm over it yields a
# second digest for free.  It is kept in RAM only (neither the prompt nor the
# hashes are ever written to disk) and is rebuilt whenever the prompt's sha256
# changes, so an identity patch or a post-compression rebuild re-arms the gate
# on the next turn.


def self_fingerprints_enabled() -> bool:
    """``HERMES_GUARD_SELF_FINGERPRINT`` — on unless explicitly switched off."""
    raw = (os.environ.get(SELF_FINGERPRINT_ENV) or "").strip().lower()
    return raw not in _FALSE_VALUES if raw else True


class SelfFingerprints:
    """One prompt's in-memory digest, tagged with the prompt hash it came from."""

    __slots__ = ("prompt_hash", "lines", "ngrams")

    def __init__(
        self, prompt_hash: str, lines: frozenset[str], ngrams: frozenset[str]
    ) -> None:
        self.prompt_hash = prompt_hash
        self.lines = lines
        self.ngrams = ngrams

    @property
    def empty(self) -> bool:
        return not (self.lines or self.ngrams)


# ── The identity segment is never protected text (2026-09-11, red item B) ──
# ``agent_identity`` is the ONE prompt section whose whole purpose is to be said
# out loud: the rule line orders the model to answer "你是谁？" with exactly
# ``我是 {name}，由 {creator} 提供``.  Fingerprinting it therefore convicts the
# model for obeying it -- on 2026-09-11 every identity answer was redacted
# (5 adjacent CJK windows, source=self/haro), 4/4, on all four managed bots.
#
# Two exits, deliberately kept separate:
#
#   * BUILD side -- :func:`strip_identity_segment` removes the segment's own
#     lines before the container fingerprints its assembled prompt, so the
#     self digest never carries them in the first place;
#   * MATCH side -- the fixed answer sentence and its cosmetic variants are
#     whitelisted, so a v1/v2 digest that Haro built WITH ``agent_identity``
#     in it (all four bots still do) cannot convict either.
#
# The whitelist covers that sentence and nothing else.  The rule line around it
# names the vendors we refuse to recite ("never call yourself Hermes"), and
# reciting *that* line is still a leak -- it stays in Haro's digest, and item 9
# of the regression (复述身份规则那一整行) still convicts through ``source=haro``.

IDENTITY_WHITELIST_ENV = "HERMES_GUARD_IDENTITY_WHITELIST"


def identity_whitelist_enabled() -> bool:
    """``HERMES_GUARD_IDENTITY_WHITELIST`` -- on unless explicitly switched off."""
    raw = (os.environ.get(IDENTITY_WHITELIST_ENV) or "").strip().lower()
    return raw not in _FALSE_VALUES if raw else True


def _active_identity():
    """The identity in force, or None.  Never raises."""
    try:
        from agent.identity_config import active_identity

        return active_identity()
    except Exception:  # noqa: BLE001 - the guard must never break a turn
        return None


def _identity_segment_keys(identity) -> frozenset:
    """Normalized keys of the identity segment's own lines."""
    try:
        from agent.identity_config import identity_prompt_lines

        return frozenset(
            key
            for key in (
                normalize_line(unicodedata.normalize("NFKC", line))
                for line in identity_prompt_lines(identity)
            )
            if key
        )
    except Exception:  # noqa: BLE001
        return frozenset()


def strip_identity_segment(text: Optional[str], identity=None) -> str:
    """``text`` with the injected identity segment's lines blanked out.

    Used on the build side only.  Lines are matched after the contract
    normalization (so spacing/case drift cannot smuggle the segment back in)
    plus, as a belt-and-braces second key, any line opening with
    :data:`~agent.identity_config.IDENTITY_RULE_PREFIX` -- that covers a reworded
    rule line whose exact text this process did not build.

    A blank line is left in place of each dropped line so the rest of the prompt
    keeps its shape; blank lines produce no fingerprints anyway.
    """
    if not isinstance(text, str) or not text:
        return text or ""
    identity = identity if identity is not None else _active_identity()
    if identity is None:
        return text
    keys = _identity_segment_keys(identity)
    try:
        from agent.identity_config import IDENTITY_RULE_PREFIX
    except Exception:  # noqa: BLE001
        IDENTITY_RULE_PREFIX = "\0"
    if not keys and not IDENTITY_RULE_PREFIX:
        return text
    out = []
    for raw in text.split("\n"):
        stripped = raw.lstrip()
        if (stripped.startswith(IDENTITY_RULE_PREFIX)
                or normalize_line(unicodedata.normalize("NFKC", raw)) in keys):
            out.append("")
        else:
            out.append(raw)
    return "\n".join(out)


_IDENTITY_WL_LOCK = threading.Lock()
#: ``(identity key, (line hashes, windows))`` -- one identity per container, so a
#: single memo slot is enough.
_IDENTITY_WL: Optional[tuple] = None


def identity_whitelist() -> tuple:
    """``(whitelisted line hashes, whitelisted 8-rune windows)`` for the answer sentence.

    Empty when no identity is configured or the feature is switched off, in
    which case the scanner behaves exactly as before.
    """
    if not identity_whitelist_enabled():
        return frozenset(), frozenset()
    identity = _active_identity()
    if identity is None:
        return frozenset(), frozenset()
    key = (identity.name, identity.creator)
    cached = _IDENTITY_WL
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        from agent.identity_config import identity_answer_variants

        variants = identity_answer_variants(identity)
    except Exception:  # noqa: BLE001
        variants = ()
    hashes: set = set()
    windows: set = set()
    for sentence in variants:
        line = normalize_line(unicodedata.normalize("NFKC", sentence))
        if not line:
            continue
        if len(line) >= MIN_LINE_RUNES:
            hashes.add(line_hash(line))
        windows.update(line_windows(line))
    built = (frozenset(hashes), frozenset(windows))
    with _IDENTITY_WL_LOCK:
        globals()["_IDENTITY_WL"] = (key, built)
    return built


def clear_identity_whitelist() -> None:
    """Drop the memoized whitelist (identity change; tests)."""
    with _IDENTITY_WL_LOCK:
        globals()["_IDENTITY_WL"] = None


_SELF_LOCK = threading.Lock()
_SELF: Optional[SelfFingerprints] = None


def register_system_prompt(text: Optional[str]) -> Optional[SelfFingerprints]:
    """Fingerprint the assembled system prompt; cached by its sha256.

    Cheap to call on every turn: an unchanged prompt costs one hash of the
    prompt bytes and returns the cached digest.  Logs the counts and the first
    eight hex of the prompt hash — never a byte of the prompt itself.
    """
    if not self_fingerprints_enabled():
        return None
    if not isinstance(text, str) or not text.strip():
        return _SELF
    prompt_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    current = _SELF
    if current is not None and current.prompt_hash == prompt_hash:
        return current
    # The identity segment is excluded before anything is hashed: it is the text
    # the model is ORDERED to say, so protecting it convicts obedience (red item
    # B, 2026-09-11).  Keyed by the hash of the ORIGINAL prompt, so the memo
    # still turns over exactly when the prompt does.
    body = strip_identity_segment(text)
    digest = SelfFingerprints(
        prompt_hash,
        # Same §6 v2 build as the generator — the two halves of the union must
        # be built by one rule, or a v2 Haro digest and a v1-ish self digest
        # would disagree about what counts as protected.
        frozenset(line_fingerprints(body, FORMAT_VERSION)),
        frozenset(ngram_fingerprints(body, FORMAT_VERSION)),
    )
    with _SELF_LOCK:
        globals()["_SELF"] = digest
    logger.info(
        "Reply guard: self fingerprints: %d lines / %d ngrams (prompt %s)",
        len(digest.lines), len(digest.ngrams), prompt_hash[:8],
    )
    return digest


def self_fingerprints() -> Optional[SelfFingerprints]:
    """The cached self digest, or None when absent or switched off."""
    digest = _SELF
    if digest is None or digest.empty or not self_fingerprints_enabled():
        return None
    return digest


def clear_self_fingerprints() -> None:
    """Drop the cached self digest (process teardown; tests)."""
    with _SELF_LOCK:
        globals()["_SELF"] = None


def note_system_prompt(text: Optional[str]) -> None:
    """Fire-and-forget hook for the prompt assembly sites. Never raises."""
    try:
        register_system_prompt(text)
    except Exception:  # noqa: BLE001 - the guard must never break a turn
        logger.debug("self fingerprint registration failed", exc_info=True)


class CombinedFingerprints:
    """Union of Haro's pushed digest and the container's self digest.

    Either half may be absent: a gateway that Haro has not pushed to still gets
    the self digest (basic protection out of the box), and switching the self
    digest off leaves exactly the previous Haro-only behaviour.
    """

    def __init__(
        self,
        haro: Optional[FingerprintStore] = None,
        own: Optional[SelfFingerprints] = None,
    ) -> None:
        self.bot_id = haro.bot_id if haro is not None else ""
        self.prompt_hash = own.prompt_hash if own is not None else ""
        self.haro_lines = haro.lines if haro is not None else frozenset()
        self.haro_ngrams = haro.ngrams if haro is not None else frozenset()
        self.self_lines = own.lines if own is not None else frozenset()
        self.self_ngrams = own.ngrams if own is not None else frozenset()
        self.lines = self.haro_lines | self.self_lines
        self.ngrams = self.haro_ngrams | self.self_ngrams

    @property
    def empty(self) -> bool:
        return not (self.lines or self.ngrams)

    def classify(self, line_hits: Iterable[str], ngram_hits: Iterable[str]) -> str:
        """``"haro"`` / ``"self"`` / ``"both"`` for the hashes that matched."""
        lines = set(line_hits)
        ngrams = set(ngram_hits)
        from_haro = bool(lines & self.haro_lines) or bool(ngrams & self.haro_ngrams)
        from_self = bool(lines & self.self_lines) or bool(ngrams & self.self_ngrams)
        if from_haro and from_self:
            return SOURCE_BOTH
        if from_self:
            return SOURCE_SELF
        if from_haro:
            return SOURCE_HARO
        return ""


_COMBINED_LOCK = threading.Lock()
_COMBINED: Optional[tuple] = None


def combined_store(path: Optional[str] = None) -> CombinedFingerprints:
    """Haro's digest (refreshed from disk) unioned with the self digest.

    Built once per (Haro digest, self digest) pair and memoized: this runs on
    every reply, and unioning two five-figure hash sets per turn is not free.
    """
    haro = store(path)
    own = self_fingerprints()
    key = (id(haro.lines), id(haro.ngrams), own.prompt_hash if own else "")
    cached = _COMBINED
    if cached is not None and cached[0] == key:
        return cached[1]
    combined = CombinedFingerprints(haro, own)
    with _COMBINED_LOCK:
        globals()["_COMBINED"] = (key, combined)
    return combined


def _has_adjacent_run(positions: Iterable[int]) -> bool:
    """True when ≥ :data:`NGRAM_RUN_THRESHOLD` matched windows sit adjacently.

    Positions are rune offsets of the matching windows in the reply's normalized
    text; a run continues while successive offsets differ by at most
    :data:`NGRAM_ADJACENT_GAP`.  Kept for callers that only have offsets;
    :func:`classify_runs` is what the scanner uses, because a run also has to
    prove it is made of something other than ASCII fragments.
    """
    ordered = sorted(set(positions))
    if len(ordered) < NGRAM_RUN_THRESHOLD:
        return False
    run = 1
    for previous, current in zip(ordered, ordered[1:]):
        run = run + 1 if current - previous <= NGRAM_ADJACENT_GAP else 1
        if run >= NGRAM_RUN_THRESHOLD:
            return True
    return False


def classify_runs(matches: dict) -> tuple[bool, bool]:
    """``(convicted, ascii_run)`` for ``{window position: window text}``.

    A run of ≥ :data:`NGRAM_RUN_THRESHOLD` adjacent matches convicts only when
    at least one of its windows :func:`window_is_convicting` — otherwise the run
    is real but made of ASCII fragments (a path, a tool name, an English
    identifier), which the 2026-09-11 regression showed every compliant answer
    produces.  Those are reported as ``ascii_run`` so the operator can watch
    them without the user losing a reply.
    """
    convicted = False
    ascii_run = False
    run: list[int] = []

    def close() -> None:
        nonlocal convicted, ascii_run
        if len(run) < NGRAM_RUN_THRESHOLD:
            return
        if any(window_is_convicting(matches[position]) for position in run):
            convicted = True
        else:
            ascii_run = True

    for position in sorted(matches):
        if run and position - run[-1] <= NGRAM_ADJACENT_GAP:
            run.append(position)
            continue
        close()
        run = [position]
    close()
    return convicted, ascii_run


class FingerprintScanner:
    """Streaming matcher: feed content deltas, ask whether the gate has tripped.

    Deltas are buffered into whole lines, because the contract normalization is
    line-scoped (trim/collapse only make sense on a complete line and 8-gram
    windows never cross a break).  An unterminated trailing line is still judged
    *provisionally* on every delta so a single-line leak is caught before the
    frame goes out; the judgement is committed at :meth:`flush`.

    Low-entropy windows are dropped before the VERDICT (and a line that is
    nothing but a path is not looked up at all), so a digest that fingerprinted
    a skill's paths and tool names cannot redact an answer for citing one — see
    :func:`is_low_entropy_window` and :attr:`ascii_run`.
    """

    def __init__(self, fingerprints: Optional[object] = None) -> None:
        self.fp = fingerprints if fingerprints is not None else combined_store()
        #: The identity answer sentence carries no evidence — see
        #: :func:`identity_whitelist`.  Resolved once per reply.
        self._wl_lines, self._wl_windows = identity_whitelist()
        self._pending = ""
        self._offset = 0  # rune offset of the pending line's start
        self._line_hits: set[str] = set()
        #: ``{window position: window text}`` — the text decides whether a run
        #: may convict (see :func:`classify_runs`).
        self._ngram_matches: dict[int, str] = {}
        self._ngram_hits: set[str] = set()
        # Hits from the not-yet-terminated line, recomputed on every delta.
        self._pending_line_hits: set[str] = set()
        self._pending_matches: dict[int, str] = {}
        self._pending_ngram_hits: set[str] = set()

    # ── verdict ────────────────────────────────────────────────────────
    def _all_matches(self) -> dict:
        merged = dict(self._ngram_matches)
        merged.update(self._pending_matches)
        return merged

    @property
    def hit_count(self) -> int:
        return len(self._line_hits | self._pending_line_hits) + len(self._all_matches())

    @property
    def tripped(self) -> bool:
        if len(self._line_hits | self._pending_line_hits) >= LINE_HIT_THRESHOLD:
            return True
        return classify_runs(convicting_matches(self._all_matches()))[0]

    @property
    def ascii_run(self) -> bool:
        """A real adjacent run that may not convict — audited, never redacted.

        Either the run is built from low-entropy windows (a cited
        ``/knowledge/…`` path: the 2026-09-11 regression's whole false-positive
        class) or it survived the filter but carries neither a non-ASCII rune
        nor two words.  The reply goes out; the operator gets one line.
        """
        if self.tripped:
            return False
        matches = self._all_matches()
        if _has_adjacent_run(matches):
            return True
        return classify_runs(convicting_matches(matches))[1]

    @property
    def source(self) -> str:
        """Which digest the matched hashes belong to (``""`` when nothing hit).

        Optional telemetry for the report body, so the operator can tell a
        pushed-digest hit from one the container fingerprinted for itself.
        """
        classify = getattr(self.fp, "classify", None)
        if classify is None:
            return SOURCE_HARO
        try:
            return classify(
                self._line_hits | self._pending_line_hits,
                self._ngram_hits | self._pending_ngram_hits,
            )
        except Exception:  # noqa: BLE001 - telemetry never breaks the redaction
            return ""

    def feed(self, text: str) -> bool:
        """Consume one delta; True once the reply is reproducing protected text."""
        if self.fp.empty or not isinstance(text, str) or not text:
            return self.tripped
        # Re-applied to the whole buffer, not the delta: a ``\r\n`` split across
        # two deltas would otherwise leave a stray ``\r`` behind.
        self._pending = (self._pending + text).replace("\r\n", "\n")
        if "\n" in self._pending:
            *complete, self._pending = self._pending.split("\n")
            for raw in complete:
                self._commit_line(raw)
        self._probe_pending()
        return self.tripped

    def flush(self) -> bool:
        """End of stream: judge the last unterminated line for real."""
        if self.fp.empty:
            return False
        raw, self._pending = self._pending, ""
        self._clear_pending()
        self._commit_line(raw)
        return self.tripped

    # ── internals ──────────────────────────────────────────────────────
    def _clear_pending(self) -> None:
        self._pending_line_hits = set()
        self._pending_matches = {}
        self._pending_ngram_hits = set()

    def _scan_line(self, raw: str) -> tuple[set[str], dict, set[str], int]:
        """``(line hits, {position: window}, gram hits, rune length)`` for one line."""
        line = normalize_line(unicodedata.normalize("NFKC", raw))
        hits: set[str] = set()
        matches: dict[int, str] = {}
        grams: set[str] = set()
        if (
            self.fp.lines
            and len(line) >= MIN_LINE_RUNES
            and not is_low_entropy_line(line)
        ):
            digest = line_hash(line)
            if digest in self.fp.lines and digest not in self._wl_lines:
                hits.add(digest)
        if self.fp.ngrams:
            for index, window in enumerate(line_windows(line)):
                if window in self._wl_windows:
                    # A window of the bot's own identity answer. Not looked up at
                    # all: it is neither evidence nor worth an audit line.
                    continue
                digest = ngram_hash(window)
                if digest in self.fp.ngrams:
                    matches[self._offset + index] = window
                    grams.add(digest)
        return hits, matches, grams, len(line)

    def _commit_line(self, raw: str) -> None:
        hits, matches, grams, length = self._scan_line(raw)
        self._line_hits |= hits
        self._ngram_matches.update(matches)
        self._ngram_hits |= grams
        # +1 for the '\n' that separated this line from the next, so windows of
        # two different lines can never look adjacent.
        self._offset += length + 1
        self._clear_pending()

    def _probe_pending(self) -> None:
        """Judge the unterminated line without committing it (bounded cost)."""
        if not self._pending or len(self._pending) > MAX_PENDING_LINE_CHARS:
            self._clear_pending()
            return
        hits, matches, grams, _ = self._scan_line(self._pending)
        self._pending_line_hits = hits
        self._pending_matches = matches
        self._pending_ngram_hits = grams


def scan_text(
    text: str, fingerprints: Optional[object] = None
) -> tuple[bool, int]:
    """Whole-text form: ``(tripped, hit_count)``."""
    scanner = FingerprintScanner(fingerprints)
    scanner.feed(text or "")
    scanner.flush()
    return scanner.tripped, scanner.hit_count


def read_sources(pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """``[(id, path), …]`` → ``[(id, text), …]`` (generator helper)."""
    out = []
    for source_id, path in pairs:
        with open(path, "r", encoding="utf-8") as fh:
            out.append((source_id, fh.read()))
    return out
