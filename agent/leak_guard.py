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
cannot judge R2 while the reply is being written.  It therefore buffers in TWO
stages instead of guessing:

  1. **Screening.**  Every reply — leak or not — is held until the accumulated
     ``content`` reaches ``SCREEN_MIN_CHARS`` (300) or ``SCREEN_MAX_SECONDS``
     (2s, monotonic, measured from the first delta), whichever comes first.
     Usage arriving early closes the window too, and so does the end of the
     stream.  R1/R3/R4 are then evaluated once on what has accumulated; no
     usage is needed for that.  The cost is bounded and paid by every reply:
     at most 300 characters or 2 seconds of deferred rendering.
  2. **Verdict.**  *Clean* → the held text is flushed in one delta and every
     later delta streams live, frame by frame.  *Suspected* → the hold runs to
     the END of the stream, where usage settles it: ``R2 && (R1||R3||R4)``
     convicts (nothing ever left), no R2 acquits (the whole buffer is flushed
     and one ``reply.audited`` line is written).  When the budget is unknown a
     suspicion is released immediately — R2 can never fire, so holding would
     only cost latency — and is likewise audited, never convicted.

The screening window is deliberately short enough that the incident text is
caught by it (the system-prompt lines sit in the FIRST 120 characters, well
inside 300) and long enough that a legitimate opening is recognisable.
``HOLD_MAX_CHARS`` / ``HOLD_MAX_SECONDS`` remain observability thresholds for
stage 2: crossing one logs the streaming-smoothness cost of a suspicion once,
and never releases the buffer.

A reply that passed screening but ends on R2 alone is **audited only** — the
user keeps it — and the audit line carries a body-free ``sample``: the shape
summary of its first 300 characters (see :func:`prefix_sample`), which is what
later threshold tuning gets to look at instead of the text itself.

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
import threading
import time
from typing import Any, Callable, NamedTuple, Optional

import logging

logger = logging.getLogger(__name__)

# ── Haro runtime API: redaction report ─────────────────────────────────
# Same env names as plugins/platforms/wecom/iac_approval.py — one operator
# configuration for every call the runtime makes back to Haro.
GUARD_API_URL_ENV = "HARO_API_URL"
GUARD_TOKEN_ENV = "HARO_RUNTIME_TOKEN"
GUARD_REPORT_PATH = "/api/assistant/runtime-api/guard/redacted"
GUARD_REPORT_TIMEOUT_SECONDS = 3.0

# The rule names Haro's endpoint accepts (400 guard_report_invalid otherwise).
RULE_REASONING_LEAK = "reasoning_leak"
RULE_PROMPT_LEAK = "prompt_leak"
# Audit-only signal: R2 fired at the end of a reply that passed screening, so
# nothing was redacted. Reported so the operator can tune the shape rules from
# the accompanying body-free ``sample``.
RULE_REASONING_LEAK_SUSPECT = "reasoning_leak_suspect"

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
# Stage 1 — the screening window every reply pays: hold until this many
# characters have accumulated or this long has passed since the first delta
# (monotonic clock), then judge the shape once.
SCREEN_MIN_CHARS = 300
SCREEN_MAX_SECONDS = 2.0
# The window whose SHAPE (never its text) is summarised into an audit sample.
SAMPLE_WINDOW = 300
# Observability thresholds for the stage-2 "suspected, waiting for usage" hold:
# crossing one logs how much live streaming the suspicion is costing (see module
# docstring for why they do not release the buffer).
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


def _char_class(ch: str) -> str:
    """A coarse category for the reply's first non-space character."""
    if not ch:
        return "empty"
    if "a" <= ch <= "z":
        return "lower_latin"
    if "A" <= ch <= "Z":
        return "upper_latin"
    if ch.isdigit():
        return "digit"
    if "一" <= ch <= "鿿":
        return "cjk"
    if ch in _R1_DANGLING_CLOSERS:
        return "closer"
    if not ch.isalnum():
        return "punct"
    return "other"


def prefix_sample(content: Optional[str]) -> dict:
    """A body-free shape summary of the reply's first ``SAMPLE_WINDOW`` chars.

    This is the ONLY thing an audit line is allowed to say about a reply that
    was not redacted: counts, ratios and rule booleans, never a character of the
    text. It exists so the R1/R3/R4 table can be tuned against the replies that
    ended on R2 alone, without the audit log becoming a copy of the leak it is
    supposed to keep off the wire.
    """
    text = content[:SAMPLE_WINDOW] if isinstance(content, str) else ""
    stripped = text.lstrip()
    first = stripped[0] if stripped else ""
    rules = evaluate_prefix(text)
    latin = sum(1 for ch in text if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    total = len(text)
    return {
        "chars": total,
        "lines": (text.count("\n") + 1) if total else 0,
        "first_char_class": _char_class(first),
        "starts_lower_latin": bool(first and "a" <= first <= "z"),
        "r1": "R1" in rules,
        "r3": "R3" in rules,
        "r4": "R4" in rules,
        "latin_ratio": round(latin / total, 3) if total else 0.0,
        "cjk_ratio": round(cjk / total, 3) if total else 0.0,
    }


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


def guard_report_url() -> Optional[str]:
    """``$HARO_API_URL`` + the redaction path, or None when unconfigured."""
    base = (os.environ.get(GUARD_API_URL_ENV) or "").strip().rstrip("/")
    if not base:
        return None
    return base + GUARD_REPORT_PATH


def guard_identity(
    session: str = "", subject: str = "", platform: str = ""
) -> dict:
    """``{session, subject, platform}`` for the report.

    The guard's own per-request fields win; anything missing falls back to the
    gateway session context via ``agent.file_safety.get_readguard_identity()``
    — the same source the read guard's audit lines use.
    """
    resolved = {
        "session": (session or "").strip(),
        "subject": (subject or "").strip(),
        "platform": (platform or "").strip(),
    }
    if all(resolved.values()):
        return resolved
    try:
        from agent.file_safety import get_readguard_identity

        fallback = get_readguard_identity()
    except Exception:  # noqa: BLE001 - reporting must never break the turn
        fallback = {}
    for key in resolved:
        if not resolved[key]:
            value = str(fallback.get(key) or "").strip()
            resolved[key] = "" if value == "unknown" else value
    return resolved


def build_guard_report(
    *,
    rule: str,
    original_len: int,
    session: str = "",
    subject: str = "",
    platform: str = "",
    reasoning_tokens: Optional[int] = None,
    budget: Optional[int] = None,
    row_id: str = "",
    sample: Optional[dict] = None,
    source: str = "",
) -> dict:
    """The report body. Carries lengths and identifiers only — never the reply.

    ``sample`` (only sent with ``reasoning_leak_suspect``) is the body-free
    shape summary from :func:`prefix_sample`.

    ``source`` (only sent with ``prompt_leak``) is ``"haro"`` / ``"self"`` /
    ``"both"`` — which digest matched.  It is an ADDITIVE optional field: a Haro
    build that predates it simply ignores the key.
    """
    identity = guard_identity(session, subject, platform)
    body = {
        "sessionId": identity["session"],
        "externalUser": identity["subject"],
        "rule": rule,
        "originalLen": int(original_len),
        "reasoningTokens": reasoning_tokens,
        "budget": budget,
        "channel": identity["platform"],
        "rowId": str(row_id or ""),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": sample,
        "source": source,
    }
    return {k: v for k, v in body.items() if v is not None and v != ""}


def post_guard_report(body: dict) -> bool:
    """POST one report to Haro. Returns True on 2xx; never raises.

    A failure is a WARN and nothing else: the redaction has already happened
    locally and the user is protected whether or not Haro hears about it.
    """
    url = guard_report_url()
    token = (os.environ.get(GUARD_TOKEN_ENV) or "").strip()
    if not url or not token:
        return False
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    try:
        try:
            import httpx

            response = httpx.post(
                url, content=payload, headers=headers,
                timeout=GUARD_REPORT_TIMEOUT_SECONDS,
            )
            status = int(response.status_code)
        except ImportError:  # pragma: no cover - httpx is a hard dep in prod
            import urllib.request

            request = urllib.request.Request(
                url, data=payload, headers=headers, method="POST"
            )
            with urllib.request.urlopen(
                request, timeout=GUARD_REPORT_TIMEOUT_SECONDS
            ) as response:
                status = int(response.status)
        if 200 <= status < 300:
            return True
        logger.warning(
            "Reply guard: Haro rejected the redaction report (status=%s rule=%s).",
            status, body.get("rule"),
        )
        return False
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.warning(
            "Reply guard: redaction report to Haro failed (%s: %s) — the local "
            "replacement stands.", type(exc).__name__, exc,
        )
        return False


def report_redaction(body: dict, *, blocking: bool = False) -> None:
    """Fire-and-forget the report on a daemon thread (``blocking`` for tests)."""
    if not guard_report_url() or not (os.environ.get(GUARD_TOKEN_ENV) or "").strip():
        return  # unconfigured deployment: reporting is simply off
    if blocking:
        post_guard_report(body)
        return
    try:
        threading.Thread(
            target=post_guard_report, args=(body,),
            name="replyguard-report", daemon=True,
        ).start()
    except Exception:  # noqa: BLE001 - thread exhaustion must not break the turn
        logger.debug("reply guard report thread failed to start", exc_info=True)


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
        row_id: str = "",
        clock: Callable[[], float] = time.monotonic,
        screen_min_chars: int = SCREEN_MIN_CHARS,
        screen_max_seconds: float = SCREEN_MAX_SECONDS,
        hold_max_chars: int = HOLD_MAX_CHARS,
        hold_max_seconds: float = HOLD_MAX_SECONDS,
    ) -> None:
        self.budget = budget
        self.platform = platform
        self.session = session
        self.subject = subject
        self.row_id = row_id
        self._clock = clock
        self._screen_min_chars = screen_min_chars
        self._screen_max_seconds = screen_max_seconds
        self._hold_max_chars = hold_max_chars
        self._hold_max_seconds = hold_max_seconds

        self.content = ""
        self.reasoning_tokens: Optional[int] = None
        self.convicted = False
        self.rules: list[str] = []
        #: R1/R3/R4 as judged once, at the end of the screening window.
        self.screen_rules: list[str] = []
        self._screened = False
        self._held: list[str] = []
        self._first_delta_at: Optional[float] = None
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
        if self._first_delta_at is None:
            self._first_delta_at = self._clock()
        return self._pump()

    # ── stage 1: the screening window ──────────────────────────────────
    def _screen_window_closed(self, *, final: bool = False) -> bool:
        """Has enough of the reply accumulated to judge its shape?

        ``final`` (end of stream) and a usage chunk both close the window early:
        in either case nothing more is coming that screening could wait for.
        """
        if final or self._usage_seen:
            return True
        if len(self.content) >= self._screen_min_chars:
            return True
        started = self._first_delta_at
        if started is None:
            return False
        return (self._clock() - started) >= self._screen_max_seconds

    def _pump(self, *, final: bool = False) -> GuardOutcome:
        """Advance the state machine and say what may go out right now."""
        if self.convicted or self._released:
            return GuardOutcome()
        if not self._screened:
            if not self._screen_window_closed(final=final):
                return GuardOutcome()  # still inside the 300-char / 2s window
            self._screened = True
            self.screen_rules = evaluate_prefix(self.content)
            if not self.screen_rules:
                return self._release()  # opening looks like a real reply
            if self.budget is None:
                # R2 can never fire without a budget, so a hold could not end in
                # a conviction — holding would only cost latency. Audited at
                # ``finish``; never convicted.
                return self._release()
        # stage 2: suspected — the hold runs until usage settles it.
        if reasoning_exhausted(self.reasoning_tokens, self.budget):
            return self._convict(evaluate_prefix(self.content) or self.screen_rules)
        if self._usage_seen or final:
            return self._release()  # acquitted: R1/R3/R4 alone never convict
        self._log_hold_cost()
        return GuardOutcome()

    def _log_hold_cost(self) -> None:
        """Warn once when the suspicion has cost more than the configured bounds."""
        if self._hold_logged:
            return
        held_chars = sum(len(part) for part in self._held)
        elapsed = self._clock() - (self._first_delta_at or self._clock())
        if held_chars < self._hold_max_chars and elapsed < self._hold_max_seconds:
            return
        self._hold_logged = True
        logger.warning(
            "Reply guard: holding a suspected reasoning leak (rules=%s held_chars=%d "
            "held_seconds=%.1f session=%s) until usage settles it; live streaming is "
            "paused for this reply.",
            "+".join(self.screen_rules), held_chars, elapsed, self.session,
        )

    def on_usage(self, usage: Any) -> GuardOutcome:
        """Feed a usage object (typically the last chunk of the stream)."""
        tokens = reasoning_tokens_from_usage(usage)
        if tokens is not None:
            self.reasoning_tokens = tokens
            self._usage_seen = True
        if self.convicted or self._released or not self._held:
            return GuardOutcome()
        return self._pump()

    def finish(self) -> GuardOutcome:
        """End of stream: flush or convict, then write the closing audit line."""
        if self.convicted:
            return GuardOutcome()
        if self._fingerprint_tripped(lambda scanner: scanner.flush()):
            return GuardOutcome(convicted=True)
        outcome = self._pump(final=True)
        if not self.convicted:
            self._audit_final()
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
        source = getattr(scanner, "source", "") or ""
        logger.warning(
            "Reply guard: outbound fingerprint hit (hits=%d source=%s session=%s "
            "platform=%s original_len=%d) — replacing the turn.",
            scanner.hit_count, source or "-", self.session, self.platform,
            len(self.content),
        )
        record_audit(
            platform=self.platform or None, session=self.session or None,
            subject=self.subject or None, rule="fingerprint",
            source=source or None,
            hit_count=scanner.hit_count, original_len=len(self.content),
        )
        self._report(RULE_PROMPT_LEAK, source=source)
        return True

    # ── reporting ──────────────────────────────────────────────────────
    def _report(self, rule: str, *, sample: Optional[dict] = None, source: str = "") -> None:
        """Tell Haro about a replaced (or merely suspected) turn.

        Best effort, body-free, off the hot path."""
        try:
            report_redaction(build_guard_report(
                rule=rule, original_len=len(self.content), session=self.session,
                subject=self.subject, platform=self.platform,
                reasoning_tokens=self.reasoning_tokens, budget=self.budget,
                row_id=self.row_id, sample=sample, source=source,
            ))
        except Exception:  # noqa: BLE001 - the replacement is what protects the user
            logger.debug("reply guard report failed", exc_info=True)

    # ── final response ─────────────────────────────────────────────────
    def final_text(self, original: Optional[str]) -> Optional[str]:
        """The text the turn should actually end with."""
        return self.redaction_text if self.convicted else original

    # ── decisions ──────────────────────────────────────────────────────
    def _release(self) -> GuardOutcome:
        self._released = True
        pending, self._held = "".join(self._held), []
        return GuardOutcome(emit=pending)

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
        self._report(RULE_REASONING_LEAK)
        return GuardOutcome(convicted=True)

    def _audit_final(self) -> None:
        """The closing audit line for a reply that was NOT redacted.

        Two shapes reach it, and neither withholds anything from the user:

        * screening suspected the opening but the evidence never came (no R2, or
          no budget at all) — the buffer was flushed, and the operator gets one
          line saying which rules fired;
        * screening was clean and R2 fired at the very end — ``reasoning_leak_suspect``:
          the reply went out frame by frame, and the line carries the body-free
          ``sample`` that the shape rules will be re-tuned against.
        """
        if self._audited or not self.content.strip():
            return
        exhausted = reasoning_exhausted(self.reasoning_tokens, self.budget)
        if self.screen_rules:
            self._audited = True
            record_audit(
                event="reply.audited", platform=self.platform or None,
                session=self.session or None, subject=self.subject or None,
                rule="+".join(self.screen_rules),
                reasoning_tokens=self.reasoning_tokens, budget=self.budget,
                original_len=len(self.content), redacted=False,
            )
            return
        if not exhausted:
            return
        self._audited = True
        sample = prefix_sample(self.content)
        logger.warning(
            "Reply guard: reasoning budget exhausted on a reply that passed screening "
            "(session=%s platform=%s original_len=%d sample=%s) — audited only, the "
            "reply stands.",
            self.session, self.platform, len(self.content), sample,
        )
        record_audit(
            event="reply.audited", platform=self.platform or None,
            session=self.session or None, subject=self.subject or None, rule="R2",
            suspect=RULE_REASONING_LEAK_SUSPECT, sample=sample,
            reasoning_tokens=self.reasoning_tokens, budget=self.budget,
            original_len=len(self.content), redacted=False,
        )
        self._report(RULE_REASONING_LEAK_SUSPECT, sample=sample)


def _make_fingerprint_scanner():
    """A scanner bound to the current digests, or None when there are none.

    The digest is the UNION of what Haro pushed and what the container
    fingerprinted from its own assembled system prompt, so a gateway Haro never
    pushed to still guards its SOUL block and role rules.
    """
    try:
        from agent.leak_fingerprints import FingerprintScanner, combined_store

        fingerprints = combined_store()
        return None if fingerprints.empty else FingerprintScanner(fingerprints)
    except Exception:
        logger.debug("fingerprint store unavailable", exc_info=True)
        return None


def resolve_row_id(agent: Any) -> str:
    """The hermes message row id of the turn being answered, best effort.

    ``_row_id`` is stamped on a persisted transcript message
    (``agent/session_persistence.py``); the most recent one identifies the turn
    Haro should attach the report to. Missing is fine — the field is optional.
    """
    try:
        messages = getattr(agent, "messages", None) or []
        for message in reversed(list(messages)[-8:]):
            if isinstance(message, dict) and isinstance(message.get("_row_id"), int):
                return str(message["_row_id"])
    except Exception:  # noqa: BLE001 - an optional field never breaks a turn
        pass
    return ""


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
            row_id=resolve_row_id(agent),
        )
    except Exception:
        logger.debug("reply guard construction failed", exc_info=True)
        return None
