"""Cheap content-sanity checks for the truncated-response continuation path.

Issue #86581: a model in a degenerate repetition loop can spend its ENTIRE
output budget echoing one fragment.  The ``finish_reason=length``
continuation path in ``conversation_loop.py`` would then retry with a
"continue, don't repeat" nudge — stitching a pathological fragment into the
final response with no content-sanity check.  In the incident behind #86581
a single turn produced a 60,698-char response delivered as 31 Discord
messages.

These helpers detect repetition-dominated fragments BEFORE the continuation
nudge is appended so the turn can abort with a clear user-facing error
(mirroring the existing ``_thinking_exhausted`` guard) instead of flooding.

The detection is deliberately conservative: only LONG verbatim repeats
(60+ chars) whose occurrences cover a majority of the fragment trip the
guard, so ordinary truncated responses (a sentence cut mid-word, a heading
repeated, code with similar-looking lines) are never blocked.
"""

from __future__ import annotations

import math

# A fragment must be at least this long before the repetition check runs at
# all.  Short truncations (a sentence cut mid-word) can trivially contain
# repeated tokens and are legitimately continued.
MIN_FRAGMENT_LENGTH = 400

# Length of the exact-repeat window.  A verbatim repeat of this many chars
# is far beyond ordinary phrasing reuse (citations, headings, similar code).
_REPEAT_WINDOW = 60

# A window that repeats at least this many times is a repetition signal,
# even for short fragments.
_MIN_REPEAT_COUNT = 5

# A fragment is "repetition-dominated" when repeated windows account for at
# least this fraction of its characters.
_DOMINANCE_RATIO = 0.5


def is_repetition_dominated(text: str) -> bool:
    """True when ``text`` is dominated by verbatim repeated fragments.

    A truncated response is "repetition-dominated" when a single 60+ char
    substring appears often enough that its occurrences cover at least half
    of the fragment.  That shape is the signature of a model repetition
    loop (issue #86581), and continuing such a fragment is pointless — the
    continuation nudge would just stitch more repeated text into the final
    response.

    Returns False for non-string / empty / short inputs (fail-open: never
    blocks a continuation the guard cannot confidently judge).
    """
    if not isinstance(text, str):
        return False
    n = len(text)
    if n < MIN_FRAGMENT_LENGTH:
        return False

    # Fast path: one normalized line duplicated often enough to cover half
    # the fragment (the most common echo shape — a repeated paragraph or
    # sentence on its own line).  Cheap, no big allocations.
    if _line_repetition_dominated(text, n):
        return True

    # General path: fixed-size exact-repeat windows, sliding one char at a
    # time.  Catches repetition loops that do not align to line boundaries.
    window = _REPEAT_WINDOW
    # A window must appear this many times for its occurrences to cover
    # >= DOMINANCE_RATIO of the fragment (and at least _MIN_REPEAT_COUNT).
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
    counts: dict[str, int] = {}
    for line in text.splitlines():
        norm = line.strip()
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
    for line, c in counts.items():
        if c >= _MIN_REPEAT_COUNT and c * len(line) >= n * _DOMINANCE_RATIO:
            return True
    return False


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
