"""Streaming reply guard: reasoning-leak egress protection (idcsre patch).

Incident 2026-09-10 (spark / WeCom, session ``20260910_082108_c2e419aa``):
a thinkcap-capped qwen3.6 exhausted its 2500-token thinking budget, vLLM
force-injected ``</think>``, and the model carried on reasoning **inside
``content``**.  The reasoning tail ended mid-sentence ("… Wait, the prompt")
and ``content`` opened with the continuation of that very sentence —
``says:\n"Be direct... No filler..."`` — i.e. verbatim system-prompt lines.
The WeCom adapter streams frames as they are generated, so those lines were
pushed to an external user before anything could judge the reply.

The tell is not the text alone (an English answer may legitimately open with
"Let me") and not the usage alone (a reply that merely used its whole budget
may still be a fine answer).  It is the CONJUNCTION:

  * **R2** (necessary) — ``usage.completion_tokens_details.reasoning_tokens``
    reached the thinking budget (vLLM reports ``budget - 1``: 2499/2500), and
    ``content`` is non-empty.  That is exactly the "budget ran out, the cap was
    force-closed, and the model kept going in the open" shape.
  * **R1 / R3 / R4** (at least one) — the first ``PREFIX_WINDOW`` characters
    look like the middle of a thought rather than the start of a reply.

``R2 && (R1 || R3 || R4)`` replaces the whole turn; **R2 alone is audited
only** (a budget-exhausted but well-formed answer must still reach the user).

Because ``usage`` normally arrives only on the LAST streaming chunk, the guard
buffers instead of guessing: once R1/R3/R4 fire on the accumulated prefix, the
outbound frames are HELD.  Usage then either convicts (nothing was leaked) or
acquits (the buffer is flushed in one delta).

The hold deliberately runs to the END of the stream rather than expiring at
``HOLD_MAX_CHARS`` / ``HOLD_MAX_SECONDS``.  Those bounds were the original
design, but on the real incident text they defeat the guard: the system-prompt
lines sit in the FIRST 120 characters, so any cap-expiry flush ships exactly
the bytes the guard exists to stop.  The caps are kept as an observability
threshold — crossing one logs the streaming-smoothness cost once — while the
suspicion itself is only ever resolved by evidence.  What a suspected reply
loses is incremental rendering, never the reply: on acquittal the whole buffer
is emitted at once and the turn finishes normally.  Holding is skipped entirely
when the thinking budget is unknown, since R2 can then never fire and a hold
could not end in a conviction.

Overriding the FINAL response text is what actually protects the user on
WeCom: ``gateway/stream_consumer.adopt_final_response`` replaces the
accumulated stream text wholesale when the final differs, and a WeCom stream
frame carries the full bubble content, so the ``finish=true`` frame overwrites
whatever intermediate frames had shown.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, NamedTuple, Optional

import logging

logger = logging.getLogger(__name__)

# Fixed operator-approved replacement for a convicted reasoning leak.
REDACTION_TEXT = "抱歉，我这次的回答生成异常，已中止。请重新发一次问题。"
# …and for a reply caught reproducing fingerprinted text (agent/leak_fingerprints.py).
FINGERPRINT_REDACTION_TEXT = "抱歉，这条回复包含不适合外发的内容，已拦截。"

# Only the opening of the reply is judged: a leak announces itself immediately
# (the model resumes the sentence its reasoning was cut off in), and scanning a
# 33k-char degenerate reply on every delta is not affordable on the hot path.
PREFIX_WINDOW = 400
# vLLM stops one token short of the cap (2499 observed against a 2500 budget);
# other servers stop exactly at it. Four tokens of slack covers both.
BUDGET_SLACK = 4
# Observability thresholds for the "suspected, waiting for usage" hold: crossing
# one logs how much live streaming the suspicion is costing (see module docstring
# for why they do not release the buffer).
HOLD_MAX_CHARS = 2000
HOLD_MAX_SECONDS = 3.0

# ── R1: the reply does not begin like a reply ──────────────────────────
# Continuation words/punctuation that only make sense attached to a preceding
# clause. The 2026-09-10 leak opened with ``says:`` (the object of "the prompt"
# from the truncated reasoning tail); the 2026-09-09 one with ``) if bkmonitor
# is working?"`` — a dangling closer.
_R1_CONTINUATION_WORDS = (
    "says:", "said:", "is ", "means ", "and ", "but ", "or ", "so ", "then ",
    "because ", "which ", "that ", "to ", "for ",
)
# A closing bracket/quote as the very first character has nothing to close.
_R1_DANGLING_CLOSERS = ")]}»”’"


def _r1_hit(prefix: str) -> bool:
    stripped = prefix.lstrip()
    if not stripped:
        return False
    first = stripped[0]
    if "a" <= first <= "z":
        return True
    if first in _R1_DANGLING_CLOSERS:
        return True
    if first in ",;":
        return True
    lowered = stripped.lower()
    return any(lowered.startswith(word) for word in _R1_CONTINUATION_WORDS)


# ── R3: inner-monologue openers (maintained word table) ────────────────
# Matched at any LINE start inside the prefix window, not only at character 0:
# a leak often spills a couple of sentences before the give-away line.
_R3_SOURCE_PATTERNS = (
    r"(?:Wait|Let's|Let me|I'll (?:output|reply|say)|I need to|Okay, so|Hmm)\b",
    r"The system prompt (?:says|mentions)",
    r"I should (?:just )?(?:reply|say|output)",
    r"Final (?:Decision|Polish|Response):",
    r"Thinking Process:",
)
_R3_RE = re.compile(r"^(?:%s)" % "|".join(_R3_SOURCE_PATTERNS), re.MULTILINE)

# ── R4: transcript replay ("User: …" / "Assistant: …" / "Turn 3:") ─────
_R4_RE = re.compile(r"(?:^|\n)(?:User: |Assistant: )|Turn \d+:")


class GuardOutcome(NamedTuple):
    """What the caller should push downstream right now.

    ``emit`` is the text to hand to the real delta sink ("" = nothing, the
    guard is holding or has convicted). ``convicted`` flips exactly once.
    """

    emit: str = ""
    convicted: bool = False


def evaluate_prefix(content: Optional[str]) -> list[str]:
    """The shape rules (``R1``/``R3``/``R4``) matched by ``content``'s opening."""
    if not isinstance(content, str) or not content.strip():
        return []
    prefix = content[:PREFIX_WINDOW]
    rules = []
    if _r1_hit(prefix):
        rules.append("R1")
    if _R3_RE.search(prefix):
        rules.append("R3")
    if _R4_RE.search(prefix):
        rules.append("R4")
    return rules


def reasoning_exhausted(reasoning_tokens: Optional[int], budget: Optional[int]) -> bool:
    """R2: the thinking budget was spent to (within ``BUDGET_SLACK`` of) the cap.

    Fail-open when either number is missing — no budget knowledge means no
    conviction, by design (a wrong redaction destroys a good answer).
    """
    if not isinstance(reasoning_tokens, int) or not isinstance(budget, int):
        return False
    if reasoning_tokens <= 0 or budget <= 0:
        return False
    return reasoning_tokens >= budget - BUDGET_SLACK


def reasoning_tokens_from_usage(usage: Any) -> Optional[int]:
    """``usage.completion_tokens_details.reasoning_tokens`` (dict or object)."""
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    if details is None and isinstance(usage, dict):
        details = usage.get("completion_tokens_details")
    if details is None:
        return None
    value = (
        details.get("reasoning_tokens")
        if isinstance(details, dict)
        else getattr(details, "reasoning_tokens", None)
    )
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def resolve_thinking_budget(agent: Any) -> Optional[int]:
    """The thinking budget in force for this request.

    Primary source is ``request_overrides.extra_body.thinking_token_budget``
    (``agent_init._apply_reasoning_max_tokens`` puts ``model.reasoning_max_tokens``
    there); ``extra_body.reasoning.max_tokens`` is the OpenRouter-shaped twin.
    ``HERMES_THINK_BUDGET_HINT`` is the last resort for deployments where a
    cap-injecting proxy (thinkcap) owns the budget and Hermes never sees it.
    """
    extra_body = {}
    try:
        overrides = getattr(agent, "request_overrides", None) or {}
        candidate = overrides.get("extra_body")
        if isinstance(candidate, dict):
            extra_body = candidate
    except Exception:
        extra_body = {}
    candidates = [extra_body.get("thinking_token_budget")]
    reasoning = extra_body.get("reasoning")
    if isinstance(reasoning, dict):
        candidates.append(reasoning.get("max_tokens"))
    candidates.append(os.environ.get("HERMES_THINK_BUDGET_HINT"))
    for raw in candidates:
        if raw is None or isinstance(raw, bool):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def guard_enabled() -> bool:
    """False when ``HERMES_REPLY_LEAK_GUARD`` is set to a falsey value."""
    raw = os.environ.get("HERMES_REPLY_LEAK_GUARD", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def audit_path() -> str:
    """``$HERMES_HOME/logs/replyguard.jsonl``."""
    home = os.environ.get("HERMES_HOME")
    if not home:
        try:
            import hermes_constants

            home = str(hermes_constants.get_hermes_home())
        except Exception:
            home = os.path.expanduser("~/.hermes")
    return os.path.join(home, "logs", "replyguard.jsonl")


def record_audit(**fields: Any) -> None:
    """Append one JSON line to the reply-guard audit log.

    NEVER carries the reply body — only its length. The whole point of the
    guard is that this text must not be written anywhere it can be read back
    out and re-sent; ``original_len`` is enough to correlate with agent logs.
    """
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": "reply.redacted"}
    event.update({k: v for k, v in fields.items() if v is not None})
    try:
        path = audit_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        logger.debug("reply guard audit write failed", exc_info=True)


class StreamLeakGuard:
    """Per-request state machine gating outbound content frames.

    Lifecycle: ``on_content_delta`` for every content delta, ``on_usage`` for
    every usage-bearing chunk, ``finish`` once at the end of the stream.
    """

    def __init__(
        self,
        *,
        budget: Optional[int] = None,
        platform: str = "",
        session: str = "",
        subject: str = "",
        clock: Callable[[], float] = time.monotonic,
        hold_max_chars: int = HOLD_MAX_CHARS,
        hold_max_seconds: float = HOLD_MAX_SECONDS,
    ) -> None:
        self.budget = budget
        self.platform = platform
        self.session = session
        self.subject = subject
        self._clock = clock
        self._hold_max_chars = hold_max_chars
        self._hold_max_seconds = hold_max_seconds

        self.content = ""
        self.reasoning_tokens: Optional[int] = None
        self.convicted = False
        self.rules: list[str] = []
        self._held: list[str] = []
        self._hold_started: Optional[float] = None
        self._released = False
        self._usage_seen = False
        self._audited = False
        self._hold_logged = False
        self.redaction_text = REDACTION_TEXT
        self._fingerprints = _make_fingerprint_scanner()

    # ── inputs ─────────────────────────────────────────────────────────
    def on_content_delta(self, text: str) -> GuardOutcome:
        """Accumulate ``text`` and say what may go out now."""
        if not isinstance(text, str) or not text:
            return GuardOutcome()
        self.content += text
        if self.convicted:
            return GuardOutcome()  # the turn is already forfeit — drop silently
        if self._fingerprint_tripped(lambda scanner: scanner.feed(text)):
            return GuardOutcome(convicted=True)
        if self._released:
            return GuardOutcome(emit=text)
        self._held.append(text)
        if self._hold_started is None:
            self._hold_started = self._clock()
        if not evaluate_prefix(self.content):
            return self._release()  # opening looks like a real reply
        if self.budget is None:
            # R2 can never fire without a budget, so a hold could not end in a
            # conviction — holding would only cost latency.
            return self._release()
        if self._usage_seen:
            return self._decide()  # usage already in hand: judge immediately
        self._log_hold_cost()
        return GuardOutcome()

    def _log_hold_cost(self) -> None:
        """Warn once when the suspicion has cost more than the configured bounds."""
        if self._hold_logged:
            return
        held_chars = sum(len(part) for part in self._held)
        elapsed = self._clock() - (self._hold_started or self._clock())
        if held_chars < self._hold_max_chars and elapsed < self._hold_max_seconds:
            return
        self._hold_logged = True
        logger.warning(
            "Reply guard: holding a suspected reasoning leak (rules=%s held_chars=%d "
            "held_seconds=%.1f session=%s) until usage settles it; live streaming is "
            "paused for this reply.",
            "+".join(evaluate_prefix(self.content)), held_chars, elapsed, self.session,
        )

    def on_usage(self, usage: Any) -> GuardOutcome:
        """Feed a usage object (typically the last chunk of the stream)."""
        tokens = reasoning_tokens_from_usage(usage)
        if tokens is not None:
            self.reasoning_tokens = tokens
            self._usage_seen = True
        if self.convicted or self._released or not self._held:
            return GuardOutcome()
        return self._decide()

    def finish(self) -> GuardOutcome:
        """End of stream: flush or convict, and audit an R2-only hit."""
        if self.convicted:
            return GuardOutcome()
        if self._fingerprint_tripped(lambda scanner: scanner.flush()):
            return GuardOutcome(convicted=True)
        outcome = self._decide(final=True) if self._held else GuardOutcome()
        if not self.convicted:
            self._audit_r2_only()
        return outcome

    # ── fingerprint gate (shares the replacement / audit path) ─────────
    def _fingerprint_tripped(self, probe) -> bool:
        """Run ``probe`` against the fingerprint scanner and convict on a hit.

        Fail-open on any error: an unreadable digest must never mute a gateway.
        """
        scanner = self._fingerprints
        if scanner is None:
            return False
        try:
            if not probe(scanner):
                return False
        except Exception:
            logger.debug("fingerprint scan failed", exc_info=True)
            return False
        self.convicted = True
        self.rules = ["fingerprint"]
        self.redaction_text = FINGERPRINT_REDACTION_TEXT
        self._held = []
        self._audited = True
        logger.warning(
            "Reply guard: outbound fingerprint hit (hits=%d session=%s platform=%s "
            "original_len=%d) — replacing the turn.",
            scanner.hit_count, self.session, self.platform, len(self.content),
        )
        record_audit(
            platform=self.platform or None, session=self.session or None,
            subject=self.subject or None, rule="fingerprint",
            hit_count=scanner.hit_count, original_len=len(self.content),
        )
        return True

    # ── final response ─────────────────────────────────────────────────
    def final_text(self, original: Optional[str]) -> Optional[str]:
        """The text the turn should actually end with."""
        return self.redaction_text if self.convicted else original

    # ── decisions ──────────────────────────────────────────────────────
    def _release(self) -> GuardOutcome:
        self._released = True
        pending, self._held = "".join(self._held), []
        return GuardOutcome(emit=pending)

    def _decide(self, *, final: bool = False) -> GuardOutcome:
        """Judge the held buffer. ``final`` = the stream is over, so an unresolved
        suspicion has to be resolved NOW, and with no R2 evidence that means release."""
        rules = evaluate_prefix(self.content)
        if not rules:
            return self._release()
        if not reasoning_exhausted(self.reasoning_tokens, self.budget):
            if self._usage_seen or final:
                return self._release()  # acquitted: R1/R3/R4 alone never convict
            return GuardOutcome()  # still waiting on usage
        return self._convict(rules)

    def _convict(self, rules: list[str]) -> GuardOutcome:
        self.convicted = True
        self.rules = ["R2", *rules]
        self._held = []
        self._audited = True
        logger.warning(
            "Reply guard: reasoning leak convicted (rules=%s reasoning_tokens=%s budget=%s "
            "session=%s platform=%s original_len=%d) — replacing the turn.",
            "+".join(self.rules), self.reasoning_tokens, self.budget,
            self.session, self.platform, len(self.content),
        )
        record_audit(
            platform=self.platform or None, session=self.session or None,
            subject=self.subject or None, rule="+".join(self.rules),
            reasoning_tokens=self.reasoning_tokens, budget=self.budget,
            original_len=len(self.content),
        )
        return GuardOutcome(convicted=True)

    def _audit_r2_only(self) -> None:
        """R2 with a well-formed reply: audit-only, the user still gets it."""
        if self._audited or not self.content.strip():
            return
        if not reasoning_exhausted(self.reasoning_tokens, self.budget):
            return
        self._audited = True
        record_audit(
            event="reply.audited", platform=self.platform or None,
            session=self.session or None, subject=self.subject or None, rule="R2",
            reasoning_tokens=self.reasoning_tokens, budget=self.budget,
            original_len=len(self.content), redacted=False,
        )


def _make_fingerprint_scanner():
    """A scanner bound to the current digest, or None when there is none."""
    try:
        from agent.leak_fingerprints import FingerprintScanner, store

        fingerprints = store()
        return None if fingerprints.empty else FingerprintScanner(fingerprints)
    except Exception:
        logger.debug("fingerprint store unavailable", exc_info=True)
        return None


def make_stream_leak_guard(agent: Any) -> Optional[StreamLeakGuard]:
    """A guard for ``agent``'s current request, or None when disabled."""
    if not guard_enabled():
        return None
    try:
        return StreamLeakGuard(
            budget=resolve_thinking_budget(agent),
            platform=str(getattr(agent, "platform", "") or ""),
            session=str(getattr(agent, "session_id", "") or ""),
            subject=str(getattr(agent, "chat_id", "") or getattr(agent, "user_id", "") or ""),
        )
    except Exception:
        logger.debug("reply guard construction failed", exc_info=True)
        return None
