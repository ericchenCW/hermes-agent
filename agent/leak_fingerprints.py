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

On-disk shape::

    {"version": 1, "botId": "…", "generatedAt": "<RFC3339>",
     "normalize": "nfkc+trim+collapse-ws+ascii-lower",
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
FORMAT_VERSION = 1
NORMALIZE_ID = "nfkc+trim+collapse-ws+ascii-lower"
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
    for byte in data:
        digest ^= byte
        digest = (digest * FNV64_PRIME) & FNV64_MASK
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


def line_fingerprints(text: str) -> list[str]:
    """Registered line hashes of ``text`` (deduplicated, ascending)."""
    return sorted(
        {
            line_hash(line)
            for line in normalize_lines(text)
            if len(line) >= MIN_LINE_RUNES
        }
    )


def ngram_fingerprints(text: str) -> list[str]:
    """8-gram hashes of ``text`` (deduplicated, ascending)."""
    out: set[str] = set()
    for line in normalize_lines(text):
        for window in line_windows(line):
            out.add(ngram_hash(window))
    return sorted(out)


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
) -> dict:
    """The on-disk digest for ``[(source_id, text), …]``. Hashes only, never text."""
    lines: set[str] = set()
    ngrams: set[str] = set()
    manifest = []
    for source_id, text in sources:
        source_lines = line_fingerprints(text)
        source_ngrams = ngram_fingerprints(text)
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
        "version": FORMAT_VERSION,
        "botId": str(bot_id or ""),
        "generatedAt": generated_at or _rfc3339_now(),
        "normalize": NORMALIZE_ID,
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
        self.ngrams: frozenset[str] = frozenset()
        self.lines: frozenset[str] = frozenset()

    @property
    def empty(self) -> bool:
        return not (self.ngrams or self.lines)

    def _clear(self) -> None:
        self.ngrams = self.lines = frozenset()
        self.bot_id = ""

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
                self.lines = frozenset(str(h) for h in (data.get("lines") or []))
                self.ngrams = frozenset(str(h) for h in (data.get("ngrams") or []))
                logger.info(
                    "Reply guard: loaded %d line and %d 8-gram fingerprints "
                    "(bot=%s) from %s",
                    len(self.lines), len(self.ngrams), self.bot_id or "-", path,
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


def _has_adjacent_run(positions: Iterable[int]) -> bool:
    """True when ≥ :data:`NGRAM_RUN_THRESHOLD` matched windows sit adjacently.

    Positions are rune offsets of the matching windows in the reply's normalized
    text; a run continues while successive offsets differ by at most
    :data:`NGRAM_ADJACENT_GAP`.
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


class FingerprintScanner:
    """Streaming matcher: feed content deltas, ask whether the gate has tripped.

    Deltas are buffered into whole lines, because the contract normalization is
    line-scoped (trim/collapse only make sense on a complete line and 8-gram
    windows never cross a break).  An unterminated trailing line is still judged
    *provisionally* on every delta so a single-line leak is caught before the
    frame goes out; the judgement is committed at :meth:`flush`.
    """

    def __init__(self, fingerprints: Optional[FingerprintStore] = None) -> None:
        self.fp = fingerprints if fingerprints is not None else store()
        self._pending = ""
        self._offset = 0  # rune offset of the pending line's start
        self._line_hits: set[str] = set()
        self._ngram_positions: set[int] = set()
        # Hits from the not-yet-terminated line, recomputed on every delta.
        self._pending_line_hits: set[str] = set()
        self._pending_positions: set[int] = set()

    @property
    def hit_count(self) -> int:
        return (
            len(self._line_hits | self._pending_line_hits)
            + len(self._ngram_positions | self._pending_positions)
        )

    @property
    def tripped(self) -> bool:
        if len(self._line_hits | self._pending_line_hits) >= LINE_HIT_THRESHOLD:
            return True
        return _has_adjacent_run(self._ngram_positions | self._pending_positions)

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
        self._pending_line_hits = set()
        self._pending_positions = set()
        self._commit_line(raw)
        return self.tripped

    # ── internals ──────────────────────────────────────────────────────
    def _scan_line(self, raw: str) -> tuple[set[str], set[int], int]:
        """``(line hits, matched window positions, rune length)`` for one raw line."""
        line = normalize_line(unicodedata.normalize("NFKC", raw))
        hits: set[str] = set()
        positions: set[int] = set()
        if self.fp.lines and len(line) >= MIN_LINE_RUNES:
            digest = line_hash(line)
            if digest in self.fp.lines:
                hits.add(digest)
        if self.fp.ngrams:
            for index, window in enumerate(line_windows(line)):
                if ngram_hash(window) in self.fp.ngrams:
                    positions.add(self._offset + index)
        return hits, positions, len(line)

    def _commit_line(self, raw: str) -> None:
        hits, positions, length = self._scan_line(raw)
        self._line_hits |= hits
        self._ngram_positions |= positions
        # +1 for the '\n' that separated this line from the next, so windows of
        # two different lines can never look adjacent.
        self._offset += length + 1
        self._pending_line_hits = set()
        self._pending_positions = set()

    def _probe_pending(self) -> None:
        """Judge the unterminated line without committing it (bounded cost)."""
        if not self._pending or len(self._pending) > MAX_PENDING_LINE_CHARS:
            self._pending_line_hits = set()
            self._pending_positions = set()
            return
        hits, positions, _ = self._scan_line(self._pending)
        self._pending_line_hits = hits
        self._pending_positions = positions


def scan_text(
    text: str, fingerprints: Optional[FingerprintStore] = None
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
