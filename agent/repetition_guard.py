"""Cheap content-sanity checks for the truncated-response continuation path.

A model in a degenerate repetition loop can spend its ENTIRE output budget echoing one fragment;
the ``finish_reason=length`` continuation would then stitch it into the final response with a
"continue" nudge (one incident: a 60k-char turn delivered as 31 Discord messages). This detects
repetition-dominated fragments BEFORE the nudge so the turn aborts with a clear error. Deliberately
conservative: only LONG verbatim repeats (60+ chars) covering a majority of the fragment trip it.
"""

from __future__ import annotations

import math
from collections import Counter

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
