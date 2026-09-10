"""Cheap content-sanity checks for the truncated-response continuation path.

A model in a degenerate repetition loop can spend its ENTIRE output budget echoing one fragment;
the ``finish_reason=length`` continuation would then stitch it into the final response with a
"continue" nudge (one incident: a 60k-char turn delivered as 31 Discord messages). This detects
repetition-dominated fragments BEFORE the nudge so the turn aborts with a clear error. Deliberately
conservative: only LONG verbatim repeats (60+ chars) covering a majority of the fragment trip it.
"""

from __future__ import annotations

import math
import re as _re
from collections import Counter, deque

# Below this length the check doesn't run: short truncations trivially
# contain repeated tokens and are legitimately continued.
MIN_FRAGMENT_LENGTH = 400
# Exact-repeat window; far beyond ordinary phrasing reuse (citations, headings, similar code).
_REPEAT_WINDOW = 60
# A window repeating at least this often is a signal even for short fragments.
_MIN_REPEAT_COUNT = 5
# "Repetition-dominated" = repeated windows cover at least this fraction.
_DOMINANCE_RATIO = 0.5


def is_repetition_dominated(text: str) -> bool:
    """True when a single 60+ char substring recurs often enough to cover at least half
    of ``text`` — the signature of a repetition loop. Fail-open for non-string/short input.

    That shape is the signature of a model repetition loop (issue #86581), and continuing such a fragment is
    pointless — the continuation nudge would just stitch more repeated text into the final response.
    """
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # Fast path: one normalized line duplicated enough to cover half the fragment (the common echo shape).
    if _line_repetition_dominated(text, n):
        return True

    # General path: fixed-size windows sliding one char at a time, catching loops that
    # don't align to line boundaries. A window must appear ``needed`` times to cover
    # >= _DOMINANCE_RATIO (and >= _MIN_REPEAT_COUNT).
    window = _REPEAT_WINDOW
    needed = max(_MIN_REPEAT_COUNT, math.ceil(n * _DOMINANCE_RATIO / window))
    counts: dict[str, int] = {}
    for i in range(n - window + 1):
        key = text[i : i + window]
        c = counts.get(key, 0) + 1
        if c >= needed:
            return True
        counts[key] = c
    return False


def _line_repetition_dominated(text: str, n: int) -> bool:
    """True when a single normalized line covers half the fragment via repeats."""
    counts = Counter(norm for norm in (line.strip() for line in text.splitlines()) if norm)
    return any(c >= _MIN_REPEAT_COUNT and c * len(line) >= n * _DOMINANCE_RATIO for line, c in counts.items())


# ── Streaming tail guard (SRE fork) ────────────────────────────────────
# The post-hoc :func:`is_repetition_dominated` check above only runs AFTER
# the provider burned the entire output budget.  A qwen3-style reasoning
# model that degenerates inside ``reasoning_content`` can spend 16k tokens
# and four minutes of wall clock on one repeated paragraph before Hermes
# ever sees the response (spark incident 2026-09-09).  The helpers below
# run DURING streaming on the accumulated tail so the request can be
# aborted as soon as the loop is unmistakable.
#
# Deliberately narrow: an exact verbatim repeat of the trailing
# ``STREAM_MIN_FRAGMENT`` characters, at least ``STREAM_MIN_REPEATS``
# times inside the trailing ``STREAM_TAIL_WINDOW`` characters.  Ordinary
# prose never repeats 120 characters verbatim three times in 2K chars.

# Trailing slice of the accumulated text the guard inspects.
STREAM_TAIL_WINDOW = 2048
# Length of the trailing fragment used as the repetition probe.
STREAM_MIN_FRAGMENT = 120
# How often that fragment must occur inside the window to trip the guard.
STREAM_MIN_REPEATS = 3
# Re-check only every N accumulated characters (str.count is cheap, but the
# streaming loop is the hottest path in the agent).
STREAM_CHECK_INTERVAL = 256


def stream_repetition_guard_enabled() -> bool:
    """False when ``HERMES_REPETITION_GUARD`` is set to a falsey value."""
    import os

    raw = os.environ.get("HERMES_REPETITION_GUARD", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def tail_repetition_detected(
    text: str,
    *,
    window: int = STREAM_TAIL_WINDOW,
    fragment: int = STREAM_MIN_FRAGMENT,
    min_repeats: int = STREAM_MIN_REPEATS,
) -> bool:
    """True when the tail of ``text`` is an obvious verbatim repeat loop.

    Takes the last ``window`` characters, uses the last ``fragment``
    characters as the probe, and counts its (non-overlapping) occurrences.
    ``str.count`` is a C-level scan, so this is O(window) per call.

    Fail-open on anything it cannot judge (non-string, too short).
    """
    if not isinstance(text, str):
        return False
    if len(text) < fragment * min_repeats:
        return False
    tail = text[-window:]
    probe = tail[-fragment:]
    if not probe.strip():
        return False
    return tail.count(probe) >= min_repeats


# ── Normalized-line loop guard (SRE fork, 2026-09-10) ──────────────────
# ``tail_repetition_detected`` needs a VERBATIM 120-char repeat, which the
# 2026-09-10 WeCom degeneration only produced after 17152 characters: the model
# looped on one short sentence, re-punctuating and re-quoting it each time
# (``"收到。随时待命。"`` / ``I'll output "收到。随时待命。"`` / ``It's fine.``),
# so the byte-exact probe kept missing.  Normalizing away whitespace,
# punctuation and digits collapses those variants onto one key and trips the
# guard at 3388 characters instead — before the model can burn 16k tokens.
#
# Deliberately narrow: FOUR occurrences of the same normalized line inside a
# forty-line window.  A genuine list or table repeats structure, not content,
# so its normalized lines stay distinct; a heading legitimately recurring three
# times in forty lines is still under the bar.

# Sliding window, in non-empty normalized lines.
LINE_LOOP_WINDOW = 40
# Occurrences of one normalized line inside that window that trip the guard.
LINE_LOOP_MIN_REPEATS = 4
# Normalized lines below this length are ignored: "1." / "是" / "```" recur
# legitimately in any structured answer.
LINE_LOOP_MIN_CHARS = 2

_LINE_NOISE_RE = _re.compile(r"\s+")
_LINE_PUNCT_RE = _re.compile(r"[^\w]+", _re.UNICODE)


def normalize_line(line: str) -> str:
    """Collapse a line onto its content identity.

    Drops whitespace and punctuation (CJK included, since ``\\w`` keeps Han
    characters but not ``。「」``) and lowercases the rest, so ``收到。`` and
    ``"收到。"`` differ only where the words differ.

    Digits are deliberately KEPT.  Stripping them collapses ``第 1 项检查通过``
    … ``第 38 项检查通过`` — an ordinary numbered checklist — onto a single key
    and trips the guard on a perfectly good answer.
    """
    if not isinstance(line, str):
        return ""
    return _LINE_PUNCT_RE.sub("", _LINE_NOISE_RE.sub("", line)).lower()


class NormalizedLineLoopDetector:
    """Streaming detector for a line repeating itself under cosmetic variation.

    Feed raw content deltas; ``feed`` returns True the first time one normalized
    line reaches ``min_repeats`` occurrences inside the trailing window.  State
    is O(window): a deque of normalized lines plus the unterminated tail.
    """

    def __init__(self, *, window: int = LINE_LOOP_WINDOW, min_repeats: int = LINE_LOOP_MIN_REPEATS,
                 min_chars: int = LINE_LOOP_MIN_CHARS) -> None:
        self._window = deque(maxlen=window)
        self._counts: Counter = Counter()
        self._min_repeats = min_repeats
        self._min_chars = min_chars
        self._pending = ""
        self.tripped = False

    def feed(self, text: str) -> bool:
        """Consume one delta; True once the loop is unmistakable (latching)."""
        if self.tripped:
            return True
        if not isinstance(text, str) or not text:
            return False
        self._pending += text
        if "\n" not in self._pending:
            return False
        *complete, self._pending = self._pending.split("\n")
        for line in complete:
            if self._observe(line):
                return True
        return False

    def _observe(self, line: str) -> bool:
        key = normalize_line(line)
        if len(key) < self._min_chars:
            return False
        if len(self._window) == self._window.maxlen:
            evicted = self._window[0]
            self._counts[evicted] -= 1
            if self._counts[evicted] <= 0:
                del self._counts[evicted]
        self._window.append(key)
        self._counts[key] += 1
        if self._counts[key] >= self._min_repeats:
            self.tripped = True
            return True
        return False


def normalized_line_loop_detected(text: str, **kwargs) -> bool:
    """Post-hoc form of :class:`NormalizedLineLoopDetector` over whole ``text``."""
    detector = NormalizedLineLoopDetector(**kwargs)
    return detector.feed(text if isinstance(text, str) else "") or detector.feed("\n")
