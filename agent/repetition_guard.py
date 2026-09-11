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
from typing import Optional

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

# Nothing shorter than this may be judged a loop at all (2026-09-11 regression:
# "你是谁？" was answered with three short variants of one identity sentence and
# the guard aborted the turn). A reply that has not yet cost 200 characters has
# not cost anything worth aborting for, and a model that really is looping will
# cross the line within milliseconds.
STREAM_MIN_REPLY_CHARS = 200


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
    total_chars: Optional[int] = None,
) -> bool:
    """True when the tail of ``text`` is an obvious verbatim repeat loop.

    Takes the last ``window`` characters, uses the last ``fragment``
    characters as the probe, and counts its (non-overlapping) occurrences.
    ``str.count`` is a C-level scan, so this is O(window) per call.

    ``total_chars`` is how many characters the reply has cost SO FAR, for
    the streaming caller that only hands over the trailing window: the
    :data:`STREAM_MIN_REPLY_CHARS` floor is about the whole reply, and
    measuring it on a pre-sliced tail would silently never bind (the tail
    reaches 2048 characters long before the floor could matter).  Defaults
    to ``len(text)`` for the whole-text callers.

    Fail-open on anything it cannot judge (non-string, too short).
    """
    if not isinstance(text, str):
        return False
    if total_chars is None:
        total_chars = len(text)
    if total_chars < STREAM_MIN_REPLY_CHARS:
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
# Deliberately narrow: FOUR occurrences of the same normalized line — of at
# least LINE_LOOP_MIN_CHARS runes, inside a forty-line window, and never before
# the reply has cost LINE_LOOP_MIN_TOTAL_CHARS characters.  A genuine list or
# table repeats structure, not content, so its normalized lines stay distinct; a
# heading legitimately recurring three times in forty lines is still under the
# bar; and a two-line identity answer (「你是谁？」, 2026-09-11) is never judged
# at all.

# Sliding window, in non-empty normalized lines.
LINE_LOOP_WINDOW = 40
# Occurrences of one normalized line inside that window that trip the guard.
LINE_LOOP_MIN_REPEATS = 4
# Normalized lines below this length are ignored: "1." / "是" / "```" / "收到"
# recur legitimately in any structured answer, and a short identity sentence
# ("我是 haro管理员") repeated by a polite model is not a degeneration.
LINE_LOOP_MIN_CHARS = 6
# …and no verdict at all before the reply has cost this many raw characters,
# for the same reason as STREAM_MIN_REPLY_CHARS above.
LINE_LOOP_MIN_TOTAL_CHARS = STREAM_MIN_REPLY_CHARS

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
                 min_chars: int = LINE_LOOP_MIN_CHARS,
                 min_total_chars: int = LINE_LOOP_MIN_TOTAL_CHARS) -> None:
        self._window = deque(maxlen=window)
        self._counts: Counter = Counter()
        self._min_repeats = min_repeats
        self._min_chars = min_chars
        self._min_total_chars = min_total_chars
        self._pending = ""
        #: Raw characters fed so far — the guard stays silent below the floor.
        self._total_chars = 0
        self.tripped = False

    def feed(self, text: str) -> bool:
        """Consume one delta; True once the loop is unmistakable (latching)."""
        if self.tripped:
            return True
        if not isinstance(text, str) or not text:
            return False
        self._total_chars += len(text)
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
        if self._total_chars < self._min_total_chars:
            # Counted, so the window is warm the moment the reply grows past the
            # floor — only the VERDICT waits.
            return False
        if self._counts[key] >= self._min_repeats:
            self.tripped = True
            return True
        return False


def normalized_line_loop_detected(text: str, **kwargs) -> bool:
    """Post-hoc form of :class:`NormalizedLineLoopDetector` over whole ``text``."""
    detector = NormalizedLineLoopDetector(**kwargs)
    return detector.feed(text if isinstance(text, str) else "") or detector.feed("\n")


# ── Reasoning-loop retry (SRE fork, 2026-09-11) ────────────────────────
# A loop inside the REASONING stream is not evidence that the model cannot
# answer: 2026-09-11 saw four aborted turns in one WeCom session and a stable
# 3/3 abort on 「写 50 字英文自我介绍」 — a request the same model answers fine
# with thinking off.  Master verdict B: on a reasoning-stream trip, stop
# consuming the stream (cancel it upstream) and re-issue the SAME request once
# with thinking switched off, instead of ending the turn.  The CONTENT stream
# keeps aborting — a loop in the delivered answer is not fixable by a retry.
#
# Scheme A (chosen): resend with the request's own thinking switch flipped off.
# Scheme B — discard the reasoning and continue the SAME stream by prefilling
# ``</think>`` — was rejected: the fork's wire is OpenAI chat-completions via
# bifrost, which exposes no prefill/continuation of an in-flight assistant turn,
# so it would mean injecting a fake assistant message and hoping the template
# glues it back — unreproducible across providers and silently corrupting the
# stored transcript.  A is one clean extra request on a path that already cost
# the user the whole turn.


def reasoning_retry_enabled() -> bool:
    """False when ``HERMES_REPETITION_REASONING_RETRY`` is set to a falsey value."""
    import os

    raw = os.environ.get("HERMES_REPETITION_REASONING_RETRY", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


#: Haro writes its per-bot routing hints into the OpenAI-standard ``user`` field,
#: e.g. ``haro;bot=b8ba5f7a;think=budget:6000`` (or ``think=inherit``).  Only the
#: ``haro;`` prefix is ours to rewrite — any other deployment's ``user`` value is
#: an opaque identifier and must be left alone.
_HARO_USER_PREFIX = "haro;"


def _haro_user_thinking_off(value):
    """Rewrite a Haro ``user`` routing string so it asks thinkcap for ``think=off``.

    Returns the new value, or ``None`` when there is nothing to change (not a
    string, not a ``haro;`` value, or already ``think=off``).

    This is the switch that actually REACHES vLLM on the production path
    hermes → bifrost → thinkcap → vLLM: bifrost drops request fields it does not
    know (``thinking_token_budget``, ``chat_template_kwargs``), so those knobs
    never leave the gateway.  ``user`` is a standard OpenAI field, so it survives,
    and thinkcap's ``parse_user_field`` reads ``think=<off|inherit|budget:N>`` out
    of it — ``off`` being the value that strips the budget and injects
    ``chat_template_kwargs.enable_thinking=false`` downstream.
    """
    if not isinstance(value, str) or not value.startswith(_HARO_USER_PREFIX):
        return None
    segments = value.split(";")
    found = False
    for i, seg in enumerate(segments):
        if seg.strip().startswith("think="):
            segments[i] = "think=off"
            found = True
    if not found:
        segments.append("think=off")
    new_value = ";".join(segments)
    return new_value if new_value != value else None


def apply_thinking_off(api_kwargs):
    """``(kwargs_copy, [switch names])`` with every thinking knob the request
    already carries turned off — or ``None`` when it carries none.

    Deliberately only touches knobs ALREADY present on the wire.  Inventing a
    provider-specific field (``extra_body.enable_thinking`` on OpenAI, say) buys
    a 400 on the retry, which is strictly worse than the abort notice it was
    meant to replace.  The one inference made is within the vLLM family: a
    request carrying ``thinking_token_budget`` is talking to a vLLM/Qwen route,
    where ``chat_template_kwargs.enable_thinking`` is the switch that actually
    stops the template from opening a ``<think>`` block, so both are set.

    Known knobs, in the shapes this fork's providers use:

    * ``extra_body.thinking_token_budget``      → ``0``   (vLLM Qwen, Haro's path)
    * ``extra_body.chat_template_kwargs.enable_thinking`` → ``False`` (vLLM)
    * ``extra_body.enable_thinking``            → ``False`` (custom OpenAI-compat)
    * ``extra_body.think``                      → ``False`` (Ollama)
    * ``extra_body.thinking``                   → ``{"type": "disabled"}`` / ``False``
    * ``extra_body.reasoning``                  → ``enabled=False, effort="none"``
    * ``extra_body.user`` / top-level ``user``  → ``think=off`` segment (Haro→thinkcap)
    * top-level ``reasoning_effort``            → ``"low"``

    The ``user`` rewrite is the one that matters on Haro's production path
    (hermes → bifrost → thinkcap → vLLM): bifrost DROPS unknown fields, so
    ``thinking_token_budget`` / ``chat_template_kwargs`` never reach the model,
    while ``user`` — a standard OpenAI field — does, and thinkcap turns its
    ``think=off`` segment into a real template-level thinking switch.  The other
    knobs are kept because flipping them costs nothing on routes that do read them.

    ``reasoning_effort`` is lowered rather than set to ``"none"``: ``"low"`` is
    accepted by every route that accepts the field at all, and this retry must
    not itself become a 400.
    """
    import copy

    if not isinstance(api_kwargs, dict):
        return None
    out = copy.deepcopy(api_kwargs)
    switches: list[str] = []
    extra = out.get("extra_body")
    if isinstance(extra, dict):
        if "thinking_token_budget" in extra:
            extra["thinking_token_budget"] = 0
            switches.append("extra_body.thinking_token_budget=0")
            ctk = extra.get("chat_template_kwargs")
            if not isinstance(ctk, dict):
                ctk = extra["chat_template_kwargs"] = {}
            if ctk.get("enable_thinking") is not False:
                ctk["enable_thinking"] = False
                switches.append("extra_body.chat_template_kwargs.enable_thinking=False")
        elif isinstance(extra.get("chat_template_kwargs"), dict) and \
                "enable_thinking" in extra["chat_template_kwargs"]:
            extra["chat_template_kwargs"]["enable_thinking"] = False
            switches.append("extra_body.chat_template_kwargs.enable_thinking=False")
        if "enable_thinking" in extra:
            extra["enable_thinking"] = False
            switches.append("extra_body.enable_thinking=False")
        if "think" in extra:
            extra["think"] = False
            switches.append("extra_body.think=False")
        if "thinking" in extra:
            extra["thinking"] = {"type": "disabled"} if isinstance(extra["thinking"], dict) else False
            switches.append("extra_body.thinking=off")
        if isinstance(extra.get("reasoning"), dict):
            extra["reasoning"].update({"enabled": False, "effort": "none"})
            extra["reasoning"].pop("max_tokens", None)
            switches.append("extra_body.reasoning=disabled")
        new_user = _haro_user_thinking_off(extra.get("user"))
        if new_user is not None:
            extra["user"] = new_user
            switches.append("user.think=off")
    new_top_user = _haro_user_thinking_off(out.get("user"))
    if new_top_user is not None:
        out["user"] = new_top_user
        if "user.think=off" not in switches:
            switches.append("user.think=off")
    if out.get("reasoning_effort") not in (None, "", "low", "none", "minimal"):
        out["reasoning_effort"] = "low"
        switches.append("reasoning_effort=low")
    return (out, switches) if switches else None
