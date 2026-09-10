"""Outbound fingerprint guard: never let the system prompt out verbatim.

The reasoning-leak rules in :mod:`agent.leak_guard` catch the *shape* of a
thought spilling into ``content``.  They cannot catch the other observed exit
(2026-09-09, two sessions): a user talks the agent into ``read_file`` on its own
``config.yaml`` and the agent obligingly quotes it back, line by line, as a
perfectly well-formed answer.  No rule about how a reply *opens* will ever see
that.

So the operator gets a second, content-addressed gate.  Haro pushes a digest of
whatever must never be echoed to
``$HERMES_HOME/replyguard/fingerprints.json`` (via WriteHome)::

    {"version": 1, "ngram": 8,
     "hashes": ["<sha1 of a normalized 8-gram>", …],
     "lines":  ["<sha1 of a normalized line>", …]}

Only hashes travel, never the protected text — the file is safe to ship to
every gateway.  Absent file ⇒ the hook is inert; the file's mtime is polled so
an updated digest takes effect without a restart.

Matching is deliberately asymmetric: **one** protected line is enough (a whole
line reproduced verbatim is not a coincidence), while character 8-grams need
**two** hits, since a single 8-gram of ordinary Chinese or a common English
phrase can collide with innocent text.

``scripts/replyguard_fingerprints.py`` builds the file from a prompt file and
is the reference for the format.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_NGRAM = 8
# Character 8-grams needed before a reply is treated as reproducing the source.
NGRAM_HIT_THRESHOLD = 2
# One whole protected line is enough.
LINE_HIT_THRESHOLD = 1
# Normalized lines shorter than this are not fingerprinted: "是"/"步骤如下" recur
# in innocent replies and would fire on everything.
MIN_LINE_CHARS = 16

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w]+", re.UNICODE)


def normalize_stream(text: str) -> str:
    """Whitespace-free, lowercased form used for character n-grams."""
    if not isinstance(text, str):
        return ""
    return _WHITESPACE_RE.sub("", text).lower()


def normalize_line(line: str) -> str:
    """Whitespace- and punctuation-free, lowercased form used for line hashes.

    Dropping punctuation is what makes a re-quoted line ("… " vs “…”) still
    match; it is the same normalization the repetition guard uses.
    """
    if not isinstance(line, str):
        return ""
    return _PUNCT_RE.sub("", _WHITESPACE_RE.sub("", line)).lower()


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def ngram_hashes(text: str, ngram: int = DEFAULT_NGRAM) -> list[str]:
    """sha1 of every ``ngram``-character window of ``normalize_stream(text)``."""
    normalized = normalize_stream(text)
    if len(normalized) < ngram:
        return []
    return [_sha1(normalized[i : i + ngram]) for i in range(len(normalized) - ngram + 1)]


def line_hashes(text: str) -> list[str]:
    """sha1 of every sufficiently long normalized line of ``text``."""
    out = []
    for raw in (text or "").splitlines():
        normalized = normalize_line(raw)
        if len(normalized) >= MIN_LINE_CHARS:
            out.append(_sha1(normalized))
    return out


def build_fingerprints(text: str, ngram: int = DEFAULT_NGRAM) -> dict:
    """The on-disk digest for ``text``. Contains hashes only, never the text."""
    return {
        "version": 1,
        "ngram": ngram,
        "hashes": sorted(set(ngram_hashes(text, ngram))),
        "lines": sorted(set(line_hashes(text))),
    }


def fingerprints_path() -> str:
    """``$HERMES_HOME/replyguard/fingerprints.json``."""
    home = os.environ.get("HERMES_HOME")
    if not home:
        try:
            import hermes_constants

            home = str(hermes_constants.get_hermes_home())
        except Exception:
            home = os.path.expanduser("~/.hermes")
    return os.path.join(home, "replyguard", "fingerprints.json")


class FingerprintStore:
    """The loaded digest, reloaded whenever the file's mtime/size changes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stamp: Optional[tuple] = None
        self._path: Optional[str] = None
        self.ngram = DEFAULT_NGRAM
        self.hashes: frozenset[str] = frozenset()
        self.lines: frozenset[str] = frozenset()

    @property
    def empty(self) -> bool:
        return not (self.hashes or self.lines)

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
                    self.hashes = self.lines = frozenset()
                    self.ngram = DEFAULT_NGRAM
                return self
            if stamp == self._stamp:
                return self
            self._stamp, self._path = stamp, path
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                ngram = int(data.get("ngram") or DEFAULT_NGRAM)
                self.ngram = ngram if ngram > 0 else DEFAULT_NGRAM
                self.hashes = frozenset(str(h) for h in (data.get("hashes") or []))
                self.lines = frozenset(str(h) for h in (data.get("lines") or []))
                logger.info(
                    "Reply guard: loaded %d n-gram and %d line fingerprints from %s",
                    len(self.hashes), len(self.lines), path,
                )
            except Exception:
                logger.warning("Reply guard: unreadable fingerprint file %s — hook stays inert.",
                               path, exc_info=True)
                self.hashes = self.lines = frozenset()
                self.ngram = DEFAULT_NGRAM
            return self


_STORE = FingerprintStore()


def store(path: Optional[str] = None) -> FingerprintStore:
    """The process-wide store, refreshed against the file on disk."""
    return _STORE.refresh(path)


class FingerprintScanner:
    """Streaming matcher: feed content deltas, ask whether the gate has tripped.

    Keeps only the trailing ``ngram - 1`` normalized characters and the current
    unterminated line, so cost is O(delta) and n-grams straddling a delta
    boundary are still seen.
    """

    def __init__(self, fingerprints: Optional[FingerprintStore] = None) -> None:
        self.fp = fingerprints if fingerprints is not None else store()
        self._tail = ""
        self._pending_line = ""
        self._ngram_hits: set[str] = set()
        self._line_hits: set[str] = set()

    @property
    def hit_count(self) -> int:
        return len(self._ngram_hits) + len(self._line_hits)

    @property
    def tripped(self) -> bool:
        return (len(self._line_hits) >= LINE_HIT_THRESHOLD
                or len(self._ngram_hits) >= NGRAM_HIT_THRESHOLD)

    def feed(self, text: str) -> bool:
        """Consume one delta; True once the reply is reproducing protected text."""
        if self.fp.empty or not isinstance(text, str) or not text:
            return self.tripped
        self._feed_ngrams(text)
        self._feed_lines(text)
        return self.tripped

    def flush(self) -> bool:
        """End of stream: judge the last unterminated line too."""
        if self.fp.empty:
            return False
        line, self._pending_line = self._pending_line, ""
        self._check_line(line)
        return self.tripped

    def _feed_ngrams(self, text: str) -> None:
        if not self.fp.hashes:
            return
        ngram = self.fp.ngram
        window = self._tail + normalize_stream(text)
        for digest in ngram_hashes(window, ngram):
            if digest in self.fp.hashes:
                self._ngram_hits.add(digest)
        self._tail = window[-(ngram - 1):] if ngram > 1 else ""

    def _feed_lines(self, text: str) -> None:
        if not self.fp.lines:
            return
        self._pending_line += text
        if "\n" not in self._pending_line:
            return
        *complete, self._pending_line = self._pending_line.split("\n")
        for line in complete:
            self._check_line(line)

    def _check_line(self, line: str) -> None:
        normalized = normalize_line(line)
        if len(normalized) < MIN_LINE_CHARS:
            return
        digest = _sha1(normalized)
        if digest in self.fp.lines:
            self._line_hits.add(digest)


def scan_text(text: str, fingerprints: Optional[FingerprintStore] = None) -> tuple[bool, int]:
    """Whole-text form: ``(tripped, hit_count)``."""
    scanner = FingerprintScanner(fingerprints)
    scanner.feed(text or "")
    scanner.flush()
    return scanner.tripped, scanner.hit_count


def iter_protected_sources(paths: Iterable[str]) -> str:
    """Concatenate fingerprint source files (generator helper)."""
    chunks = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            chunks.append(fh.read())
    return "\n".join(chunks)
