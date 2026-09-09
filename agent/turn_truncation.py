"""Truncation recovery (``finish_reason == "length"``) for the conversation turn loop.

Handles thinking-budget exhaustion, repetition-dominated truncation, content-filter stream
stalls escalated to the fallback chain, text continuation nudges (up to 4, with the ceiling
exit that drops the fragment trail), truncated tool-call retries with max_tokens boosts, and
the final roll-back. Nothing here imports ``agent.conversation_loop`` at module level
(cycle); loop-internal helpers are imported lazily so tests patching them keep working.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent.error_classifier import FailoverReason
from agent.message_metadata import append_message
from agent.message_sanitization import close_interrupted_tool_sequence
from agent.repetition_guard import is_repetition_dominated
from agent.turn_api_call import stop_thinking_spinner
from agent.turn_retry_state import TurnRetryState
from hermes_constants import PARTIAL_STREAM_STUB_ID

logger = logging.getLogger("agent.conversation_loop")

_CONTINUABLE_MODES = {"chat_completions", "bedrock_converse", "anthropic_messages"}
_THINK_TAG_RE = re.compile(r'<(?:think|thinking|reasoning|REASONING_SCRATCHPAD)[^>]*>', re.IGNORECASE)
_TRUNCATED_FINAL = "Response truncated due to output length limit"
_FIRST_TRUNCATED_FINAL = "First response truncated due to output length limit"
# #106260: a stream that died on a context-overflow error after partial delivery must not seed a
# continuation — the transcript already cannot fit, and appending the partial stub grows every
# later request into the same overflow. End the turn via the recovery contract instead.
_CONTEXT_OVERFLOW_PARTIAL_FINAL = (
    "The request no longer fits the model's context window, so the partial "
    "response was not continued. Continue in a fresh session (/new; gateway "
    "chats are reset automatically)."
)

_THINKING_EXHAUSTED = (
    "💭 Reasoning exhausted the output token budget — no visible response was produced.",
    "⚠️ **Thinking Budget Exhausted**\n\nThe model used all its output tokens on reasoning "
    "and had none left for the actual response.\n\nTo fix this:\n"
    "→ Lower reasoning effort: `/reasoning low` or `/reasoning minimal`\n"
    "→ Or switch to a larger/non-reasoning model with `/model`",
    "Model used all output tokens on reasoning with none left "
    "for the response. Try lowering reasoning effort or increasing max_tokens.",
)
_REPETITION_DOMINATED = (
    "🔁 Response dominated by repeated text — stopping instead of continuing a degenerate response.",
    "⚠️ **Response Stopped — Repetition Detected**\n\nThe model fell into a repetition loop while "
    "writing this response, so continuing would only produce more repeated text. The partial response "
    "was discarded.\n\n→ Switch to a different model with `/model`\n"
    "→ Or resend your message (your conversation history is preserved)",
    "Model output entered a repetition loop and was truncated mid-loop; refusing to continue a "
    "degenerate response.",
)
_CEILING_NO_TEXT = (
    "⚠️ **No visible answer was produced.** The model hit its output-token limit on every "
    "continuation attempt — its reasoning consumed the entire budget each time.\n\nTo fix this:\n"
    "→ Lower reasoning effort: `/reasoning low` or `/reasoning none`\n→ Or raise max_tokens for this model"
)


def normalize_response_for_agent(agent: Any, response: Any) -> Any:
    """One OpenAI-style message from any transport; Anthropic strips the OAuth tool prefix."""
    if agent.api_mode == "anthropic_messages":
        return agent._get_transport().normalize_response(
            response, strip_tool_prefix=agent._is_anthropic_oauth
        )
    return agent._get_transport().normalize_response(response)


def partial_result(
    messages: List[Dict[str, Any]], api_call_count: int, final_response: str,
    error: Optional[str] = None, *, failed: bool = False, compression_exhausted: bool = False,
    error_detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Typed incomplete-turn result (``partial`` unless ``failed``); ``error`` defaults to
    ``final_response``. ``compression_exhausted`` carries the #98722 typed bit the gateway
    consumes to reset/move future input to a clean session (see run_turn.py).

    idcsre patch: ``error_detail`` carries the English diagnostic when ``error`` holds the
    operator-facing Chinese notice, so log scrapers and tests keep a stable string."""
    result = {
        "final_response": final_response,
        "messages": messages,
        "api_calls": api_call_count,
        "completed": False,
        ("failed" if failed else "partial"): True,
        "error": final_response if error is None else error,
    }
    if compression_exhausted:
        result["compression_exhausted"] = True
    if error_detail:
        result["error_detail"] = error_detail
    return result


@dataclass
class TruncationVerdict:
    """Outcome of ``recover_from_truncation``.

    ``action``: ``"return"`` (end the turn with ``result``), ``"break"`` (a
    ``_retry.restart_with_*`` flag is set — restart the API call), ``"continue"``
    (re-issue the same call immediately) or ``"fallthrough"`` (unreachable in practice:
    every path exits, kept for the contract). The remaining fields are the loop locals
    the handler may have rebound."""

    action: str
    result: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]
    length_continue_retries: int
    truncated_response_parts: List[str]
    truncated_tool_call_retries: int
    retry_count: int
    compression_attempts: int


@dataclass(kw_only=True)
class _Trunc(TruncationVerdict):
    """Working state for the truncation phases — the verdict itself, plus the read-only
    call context; phases mutate the loop-local fields and ``done()`` stamps the action."""

    agent: Any
    response: Any
    finish_reason: str
    conversation_history: Any
    api_call_count: int
    effective_task_id: Any
    current_turn_user_idx: Any
    action: str = "fallthrough"
    result: Optional[Dict[str, Any]] = None
    # idcsre patch: normalized response bits stashed for the readguard-style audit lines.
    _trunc_content: Any = None
    _trunc_has_tool_calls: bool = False

    def done(self, action: str, result: Optional[Dict[str, Any]] = None) -> TruncationVerdict:
        self.action, self.result = action, result
        return self

    def end_turn(
        self, final_response: str, error: Optional[str] = None, *,
        result_messages: Optional[List[Dict[str, Any]]] = None, cleanup: bool = True,
        failed: bool = False, compression_exhausted: bool = False,
        error_detail: Optional[str] = None,
    ) -> TruncationVerdict:
        """Persist and end the turn as partial (or ``failed``).

        ``compression_exhausted`` forwards the #98722 typed bit so the gateway can
        move future input off a bloated session (run_turn.py consumes it).
        """
        agent = self.agent
        if cleanup:
            agent._cleanup_task_resources(self.effective_task_id)
        agent._persist_session(self.messages, self.conversation_history)
        return self.done("return", partial_result(
            self.messages if result_messages is None else result_messages, self.api_call_count,
            final_response, error, failed=failed, compression_exhausted=compression_exhausted,
            error_detail=error_detail,
        ))

    @property
    def is_stub(self) -> bool:
        return getattr(self.response, "id", "") == PARTIAL_STREAM_STUB_ID


def _log_length_event(st: Any, action: str, *, attempt: Any = None, max_attempts: Any = None) -> None:
    """idcsre patch — one WARNING line per decision on the finish_reason=length path.

    The whole truncation / continuation / give-up chain used to be silent, so a production
    incident could only be reconstructed from the gateway's own request log."""
    from agent.conversation_loop import _reasoning_tokens_from_usage, log_length_truncation

    agent = st.agent
    content = getattr(st, "_trunc_content", None)
    log_length_truncation(
        session_id=getattr(agent, "session_id", None),
        usage=getattr(st.response, "usage", None),
        reasoning_tokens=_reasoning_tokens_from_usage(st.response),
        has_content=bool(content and str(content).strip()),
        has_tool_calls=bool(getattr(st, "_trunc_has_tool_calls", False)),
        attempt=st.length_continue_retries if attempt is None else attempt,
        max_attempts=max_attempts,
        action=action,
    )


def _exhausted_budget(agent: Any, response: Any) -> Optional[int]:
    """The output budget quoted in the thinking-exhaustion notice."""
    usage = getattr(response, "usage", None)
    return getattr(usage, "completion_tokens", None) or getattr(agent, "max_tokens", None)


def _drop_continuation_trail(st: _Trunc) -> None:
    """Drop this turn's continuation fragments + unanswered nudges.

    Leaving them behind made every later turn re-truncate against the same dead weight."""
    idx = st.current_turn_user_idx
    _turn_start = idx + 1 if isinstance(idx, int) and idx >= 0 else 0
    st.messages[_turn_start:] = [
        m for m in st.messages[_turn_start:]
        if not (isinstance(m, dict) and (
            m.get("_length_continuation_fragment") or m.get("_length_continuation_nudge")
        ))
    ]


def _abort_reason(
    agent: Any, content: Any, has_tool_calls: bool, response: Any = None,
    *, length_continue_retries: int = 0,
) -> Optional[tuple]:
    """``(vprint, user response, error)`` when continuation must NOT be attempted:
    the stream repetition guard already aborted the request, thinking exhausted the budget
    (reasoning blocks with no visible text after them), or a repetition loop burned the budget
    on one fragment (reasoning stripped first).

    idcsre patch: two changes to the thinking-exhaustion arm.  (a) Side-channel reasoning —
    vLLM ``--reasoning-parser qwen3``, DeepSeek, OpenRouter — never emits ``<think>`` tags and
    returns ``content: null``, so the tag test alone missed the most common exhaustion shape in
    production; ``reasoning_content`` / ``reasoning`` / ``reasoning_details`` and
    ``usage.completion_tokens_details.reasoning_tokens`` are now first-class evidence.  Note this
    makes ``content=None`` with reasoning evidence an abort rather than a normal truncation.
    (b) The operator-facing text is Chinese (the English diagnostic stays in ``error``).
    (c) Compromise semantics (master verdict 8): the side-channel arm from (a) only fires once a
    continuation request has already gone out (``length_continue_retries > 0``).  Upstream reads
    that shape as a normal truncation and answers it with the one-shot reasoning-off continuation,
    so the FIRST occurrence falls through to ``_continue_text`` and gets exactly that retry; a
    SECOND one is declared exhaustion.  Upstream's own arm — inline ``<think>`` tags with no text
    after them — keeps aborting immediately, exactly as on main."""
    from agent.conversation_loop import (
        _reasoning_tokens_from_usage, _repetition_abort_message, _response_reasoning_text,
        _thinking_exhausted_message, _thinking_exhausted_signal,
    )

    if has_tool_calls:
        return None
    # The stream guard already cut the request short: the partial output is a degenerate loop,
    # so neither continuing it nor keeping it is useful.
    if response is not None and getattr(response, "_repetition_aborted", False):
        return (
            "🔁 Aborted mid-stream — model output entered a repetition loop.",
            _repetition_abort_message(),
            "Model output entered a repetition loop mid-stream; the request was aborted "
            "before the output budget was exhausted.",
        )
    reasoning_tokens = _reasoning_tokens_from_usage(response)
    has_think_tags = bool(content and _THINK_TAG_RE.search(content))
    if _thinking_exhausted_signal(
        has_tool_calls=has_tool_calls,
        content=content,
        has_think_tags=has_think_tags,
        has_content_after_think=agent._has_content_after_think_block,
        reasoning_text=_response_reasoning_text(response),
        reasoning_tokens=reasoning_tokens,
    ):
        if not has_think_tags and length_continue_retries <= 0:
            # Side-channel shape on its FIRST occurrence: upstream treats this as a normal
            # truncation and answers it with the one-shot reasoning-off continuation, so fall
            # through to _continue_text and let that retry happen. Upstream's own arm (inline
            # <think> tags with nothing after them) still aborts on the spot, unchanged.
            return None
        return (_THINKING_EXHAUSTED[0], _thinking_exhausted_message(_exhausted_budget(agent, response)),
                _THINKING_EXHAUSTED[2])
    visible = agent._strip_think_blocks(content) if isinstance(content, str) else content
    if visible and is_repetition_dominated(visible):
        return _REPETITION_DOMINATED
    return None


def _content_filter_fallback(st: _Trunc, _retry: TurnRetryState) -> Optional[TruncationVerdict]:
    """Content-filter stream stall → fallback. ``_content_filter_terminated`` is
    content-deterministic, so escalate before retrying the primary; without a fallback
    fall through to normal continuation (best-effort, may loop)."""
    agent = st.agent
    if not (
        getattr(st.response, "_content_filter_terminated", False)
        and agent._fallback_index < len(agent._fallback_chain)
    ):
        return None
    agent._vprint(
        f"{agent.log_prefix}🛡️  Content filter terminated stream — activating fallback provider...",
        force=True,
    )
    agent._emit_status("Content filter terminated stream; switching to fallback...")
    if agent._try_activate_fallback():
        # Roll partial content back to the last clean turn so the fallback gets a
        # coherent continuation point; unmark survivors (their text left the partial).
        if st.truncated_response_parts:
            st.messages = agent._get_messages_up_to_last_assistant(st.messages)
        for _frag in st.messages:
            if isinstance(_frag, dict):
                _frag.pop("_length_continuation_fragment", None)
                _frag.pop("_length_continuation_nudge", None)
        agent._session_messages = st.messages
        st.length_continue_retries = 0
        st.truncated_response_parts = []
        st.retry_count = 0
        st.compression_attempts = 0
        _retry.primary_recovery_attempted = False
        _retry.restart_with_rebuilt_messages = True
        return st.done("break")
    agent._vprint(
        f"{agent.log_prefix}⚠️  No fallback provider configured — retrying with same provider "
        f"(may re-hit filter)...",
        force=True,
    )
    return None


def _continue_text(st: _Trunc, _retry: TurnRetryState, assistant_message: Any) -> TruncationVerdict:
    """Text truncation (no tool calls): append the fragment + a continuation nudge (up to
    4), then the ceiling exit that drops the fragment trail and keeps the stitched partial.
    Never appends an interim assistant row with NO visible content — strict providers
    reject it with 400 — only the nudge."""
    from agent.conversation_loop import (
        _get_continuation_prompt, _join_truncated_parts, _length_continuation_max_attempts,
        _length_continuation_worthwhile, _length_giveup_message, _thinking_exhausted_message,
    )

    agent = st.agent
    messages = st.messages
    # idcsre patch: the counter must only advance when a continuation request is actually
    # ISSUED. It used to advance on every truncation, so a turn that never continued still
    # reported one attempt and the give-up text claimed four. ``max_attempts`` is the number of
    # such sends (default 3, HERMES_LENGTH_CONTINUATION_ATTEMPTS overrides), which keeps the
    # per-turn request count identical to upstream's ``n < 4`` on the old counter.
    max_attempts = _length_continuation_max_attempts()
    n = st.length_continue_retries
    _interim_content = getattr(assistant_message, "content", None)
    _reasoning_only = not _interim_content and not st.is_stub
    if _reasoning_only:
        # Thinking-only truncation: continuing with thinking ON re-burns the budget, so the
        # (single) continuation upstream allows goes out with reasoning disabled.
        agent._ephemeral_reasoning_off = True
    if _interim_content:
        interim_msg = agent._build_assistant_message(assistant_message, st.finish_reason)
        interim_msg["_length_continuation_fragment"] = True  # ceiling exit drops these
        append_message(messages, interim_msg)
        st.truncated_response_parts.append(_interim_content)

    # idcsre patch (compromise semantics): a reasoning-only truncation is continued exactly once
    # — with reasoning off — and a second one is declared thinking-exhaustion below rather than
    # burning another budget (see _length_continuation_worthwhile). The ceiling exit further down
    # keeps whatever is stitched so far.
    if n < max_attempts and _length_continuation_worthwhile(
        assistant_message, st.truncated_response_parts, retries=n
    ):
        st.length_continue_retries += 1
        n = st.length_continue_retries
        _log_length_event(st, "continue", attempt=n, max_attempts=max_attempts)
        _dropped_tools = getattr(st.response, "_dropped_tool_names", None)
        if st.is_stub and _dropped_tools:
            agent._vprint(
                f"{agent.log_prefix}↻ Stream interrupted mid "
                f"tool-call ({', '.join(_dropped_tools[:3])}) — requesting chunked retry "
                f"({n}/{max_attempts})..."
            )
        elif st.is_stub:
            agent._vprint(
                f"{agent.log_prefix}↻ Stream interrupted — requesting continuation ({n}/{max_attempts})...")
        else:
            agent._vprint(f"{agent.log_prefix}↻ Requesting continuation ({n}/{max_attempts})...")
        append_message(messages, {
            "role": "user", "content": _get_continuation_prompt(st.is_stub, _dropped_tools),
            "_length_continuation_nudge": True,
        })
        agent._session_messages = messages
        _retry.restart_with_length_continuation = True
        return st.done("break")

    if _reasoning_only and n > 0 and not st.truncated_response_parts:
        # Second consecutive thinking-only truncation, and the reasoning-off continuation was
        # already spent: the model cannot get past its own reasoning this turn. Abort with the
        # fork's Chinese notice instead of spending the remaining continuation budget.
        _log_length_event(st, "thinking_exhausted", attempt=n, max_attempts=max_attempts)
        agent._ephemeral_reasoning_off = False
        agent._vprint(f"{agent.log_prefix}{_THINKING_EXHAUSTED[0]}", force=True)
        _drop_continuation_trail(st)
        agent._session_messages = messages
        return st.end_turn(
            _thinking_exhausted_message(_exhausted_budget(agent, st.response)),
            _THINKING_EXHAUSTED[2],
            error_detail=(
                "Reasoning-only truncation repeated after the reasoning-off continuation "
                f"(attempt {n}/{max_attempts})"
            ),
        )

    _log_length_event(st, "give_up", attempt=n, max_attempts=max_attempts)
    _giveup_message = _length_giveup_message(n, max_attempts)
    partial_response = agent._strip_think_blocks(_join_truncated_parts(st.truncated_response_parts)).strip()
    # The one-shot reasoning-off override must not leak into the next turn.
    agent._ephemeral_reasoning_off = False
    agent._vprint(
        f"{agent.log_prefix}⚠️  Response still truncated after {n} continuation attempt(s) — "
        + ("keeping the partial response received so far." if partial_response
           else "no visible text was produced."),
        force=True,
    )
    # Unanswered continue nudges made every later turn re-truncate: drop the trail.
    _drop_continuation_trail(st)
    if partial_response:
        append_message(messages, {
            "role": "assistant", "content": partial_response, "finish_reason": "length"
        })
    agent._session_messages = messages
    return st.end_turn(
        partial_response or _giveup_message,
        _giveup_message,
        error_detail=(
            f"Response remained truncated after {n} continuation attempt(s) "
            f"(ceiling {max_attempts})"
        ),
    )


def _retry_truncated_tool_call(st: _Trunc, api_kwargs: Any) -> TruncationVerdict:
    """Truncated tool call: re-run the same call (up to 4×) with a boosted max_tokens —
    a real output-cap truncation needs it, harmless for a network stall — else refuse to
    execute incomplete arguments."""
    from agent.conversation_loop import (
        TRUNCATED_TOOL_CALL_MAX_RETRIES, _truncated_tool_call_message,
    )

    agent = st.agent
    # idcsre patch: named constant, separate from the text-continuation ceiling — that counter now
    # counts continuation REQUESTS, this one always counted whole-turn replays.
    max_retries = TRUNCATED_TOOL_CALL_MAX_RETRIES
    if st.truncated_tool_call_retries < max_retries:
        st.truncated_tool_call_retries += 1
        n = st.truncated_tool_call_retries
        if st.is_stub:
            agent._buffer_vprint(f"⚠️  Stream interrupted mid tool-call — retrying ({n}/{max_retries})...")
        else:
            agent._buffer_vprint(
                f"⚠️  Truncated tool call detected — retrying API call ({n}/{max_retries})...")
        _tc_boost = (agent.max_tokens if agent.max_tokens else 4096) * (2 ** n)
        _tc_requested_cap = agent._requested_output_cap_from_api_kwargs(api_kwargs)
        if _tc_requested_cap is not None:
            _tc_boost = max(_tc_boost, _tc_requested_cap)
        agent._ephemeral_max_output_tokens = min(_tc_boost, max(32768, _tc_requested_cap or 0))
        return st.done("continue")  # don't append the broken response
    agent._flush_status_buffer()
    if st.is_stub:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Stream kept dropping mid tool-call after {max_retries} retries "
            "— the action was not executed.",
            force=True,
        )
        _detail = "Stream repeatedly dropped mid tool-call (network); the tool was not executed"
    else:
        agent._vprint(
            f"{agent.log_prefix}⚠️  Truncated tool call response detected again — refusing to execute incomplete tool arguments.",
            force=True,
        )
        _detail = _TRUNCATED_FINAL
    _log_length_event(st, "give_up_tool_call", attempt=st.truncated_tool_call_retries,
                      max_attempts=max_retries)
    _final_response = _truncated_tool_call_message(st.is_stub)
    agent._cleanup_task_resources(st.effective_task_id)
    # Prior tool batches can leave a tool-result tail; this path never reaches finalize_turn.
    close_interrupted_tool_sequence(st.messages, _final_response)
    return st.end_turn(_final_response, cleanup=False, error_detail=_detail)


def recover_from_truncation(
    agent: Any, response: Any, finish_reason: str, _retry: TurnRetryState, *,
    messages: List[Dict[str, Any]], conversation_history: Any, api_kwargs: Any, api_call_count: int,
    effective_task_id: Any, current_turn_user_idx: Any, length_continue_retries: int,
    truncated_response_parts: List[str], truncated_tool_call_retries: int, retry_count: int,
    compression_attempts: int,
) -> TruncationVerdict:
    """Recover from a truncated response. Order is load-bearing: thinking exhaustion and
    repetition abort BEFORE any continuation; a content-filter stall escalates to the
    fallback chain BEFORE the primary is retried; text continuation (no tool calls) then
    truncated tool-call retry; finally roll back to the last complete assistant turn."""
    st = _Trunc(
        agent=agent, response=response, finish_reason=finish_reason,
        conversation_history=conversation_history, api_call_count=api_call_count,
        effective_task_id=effective_task_id, current_turn_user_idx=current_turn_user_idx,
        messages=messages, length_continue_retries=length_continue_retries,
        truncated_response_parts=truncated_response_parts,
        truncated_tool_call_retries=truncated_tool_call_retries, retry_count=retry_count,
        compression_attempts=compression_attempts,
    )
    agent._vprint(
        f"{agent.log_prefix}⚠️  Response truncated — stream ended before completion"
        if st.is_stub else
        f"{agent.log_prefix}⚠️  Response truncated (finish_reason='length') - model hit max output tokens",
        force=True,
    )

    # #106260: a context-overflow error after partial delivery must not seed a
    # continuation. _partial_stream_stub marks such stubs _overflow_terminal and
    # leaves content empty; continuing would only re-send a larger request into
    # the same overflow. The stub path never raises, so this class never reached
    # recover_from_overflow's compress-and-retry on main either — ending the turn
    # replaces a growth loop, not a compression attempt.
    if getattr(st.response, "_overflow_terminal", False):
        agent._flush_status_buffer()
        agent._vprint(
            f"{agent.log_prefix}⚠️ Stream ended on a context-overflow error after "
            "partial delivery — not continuing (the request no longer fits the model's "
            "context window).",
            force=True,
        )
        # Prior tool batches can leave a tool-result tail; this path never reaches
        # finalize_turn (same as the truncated-tool-call terminal above).
        close_interrupted_tool_sequence(st.messages, _CONTEXT_OVERFLOW_PARTIAL_FINAL)
        # Carry the #98722 typed exhaustion bit so the gateway resets/moves future
        # input to a clean session instead of leaving this bloated one authoritative
        # for the next turn.
        return st.end_turn(
            _CONTEXT_OVERFLOW_PARTIAL_FINAL,
            error=_CONTEXT_OVERFLOW_PARTIAL_FINAL,
            failed=True,
            compression_exhausted=True,
        )

    _trunc_msg = normalize_response_for_agent(agent, response)
    _trunc_content = getattr(_trunc_msg, "content", None) if _trunc_msg else None
    _trunc_has_tool_calls = bool(getattr(_trunc_msg, "tool_calls", None)) if _trunc_msg else False
    # idcsre patch: carried on the working state so the audit lines below can report them.
    st._trunc_content = _trunc_content
    st._trunc_has_tool_calls = _trunc_has_tool_calls

    abort = _abort_reason(
        agent, _trunc_content, _trunc_has_tool_calls, response,
        length_continue_retries=st.length_continue_retries,
    )
    if abort is not None:
        line, user_response, error = abort
        if st.length_continue_retries > 0:
            # A continuation nudge already went out this turn; don't leave it (or the
            # fragments) in the transcript for later turns to re-truncate against.
            _drop_continuation_trail(st)
            agent._ephemeral_reasoning_off = False
            agent._session_messages = st.messages
        _log_length_event(
            st,
            "repetition_abort" if getattr(response, "_repetition_aborted", False)
            else ("thinking_exhausted" if user_response is not _REPETITION_DOMINATED[1]
                  else "repetition_dominated"),
        )
        agent._vprint(f"{agent.log_prefix}{line}", force=True)
        return st.end_turn(user_response, error)
    _log_length_event(st, "truncated")

    if agent.api_mode in _CONTINUABLE_MODES:
        cf = _content_filter_fallback(st, _retry)
        if cf is not None:
            return cf
        if _trunc_msg is not None:
            if not _trunc_has_tool_calls:
                return _continue_text(st, _retry, _trunc_msg)
            return _retry_truncated_tool_call(st, api_kwargs)

    if len(messages) > 1:
        agent._vprint(f"{agent.log_prefix}   ⏪ Rolling back to last complete assistant turn")
        return st.end_turn(
            _TRUNCATED_FINAL, result_messages=agent._get_messages_up_to_last_assistant(messages)
        )
    # First message was truncated - mark as failed
    agent._flush_status_buffer()
    agent._vprint(f"{agent.log_prefix}❌ First response truncated - cannot recover", force=True)
    return st.end_turn(_FIRST_TRUNCATED_FINAL, cleanup=False, failed=True)


_CODEX_REPLAY_KEYS = (
    "content", "reasoning", "reasoning_content", "reasoning_details",
    "codex_reasoning_items", "codex_message_items",
)


def continue_codex_incomplete(
    agent: Any, assistant_message: Any, finish_reason: str, *, messages: List[Dict[str, Any]],
    conversation_history: Any, api_call_count: int,
) -> Optional[Dict[str, Any]]:
    """Codex Responses ``status=incomplete`` continuation (max 3 per turn).

    Appends the interim assistant message (deduped on visible content only — opaque
    provider state drifts per continuation; ``codex_reasoning_items`` are merged, not
    overwritten, because the earlier response holds the only native-compaction
    checkpoint) and, when a bare retry would be byte-identical, a user-role nudge — only
    after an assistant row, to preserve role alternation. Returns ``None`` to continue
    the turn loop, or the terminal ``partial`` result once retries are exhausted."""
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE

    agent._codex_incomplete_retries += 1
    n = agent._codex_incomplete_retries

    interim_msg = agent._build_assistant_message(assistant_message, finish_reason)
    interim_has_content = bool((interim_msg.get("content") or "").strip())
    _reasoning = interim_msg.get("reasoning")
    interim_has_reasoning = isinstance(_reasoning, str) and bool(_reasoning.strip())
    interim_has_codex_reasoning = bool(interim_msg.get("codex_reasoning_items"))
    interim_has_codex_message_items = bool(interim_msg.get("codex_message_items"))

    if interim_has_content or interim_has_reasoning or interim_has_codex_reasoning or interim_has_codex_message_items:
        last_msg = messages[-1] if messages else None
        last_is_dict = isinstance(last_msg, dict)
        last_interim_visible = agent._interim_assistant_visible_text(last_msg) if last_is_dict else ""
        current_interim_visible = agent._interim_assistant_visible_text(interim_msg)
        if last_interim_visible or current_interim_visible:
            same_visible_output = last_interim_visible == current_interim_visible
        else:
            # Neither has text eligible for interim delivery: compare raw content+reasoning.
            same_visible_output = last_is_dict and (
                (last_msg.get("content") or "") == (interim_msg.get("content") or "")
                and (last_msg.get("reasoning") or "") == (interim_msg.get("reasoning") or "")
            )
        if (
            last_is_dict
            and last_msg.get("role") == "assistant"
            and last_msg.get("finish_reason") == "incomplete"
            and same_visible_output
        ):
            # Duplicate: refresh replay state in place, no re-emitted commentary.
            for _key in _CODEX_REPLAY_KEYS:
                if _key not in interim_msg:
                    continue
                if _key == "codex_reasoning_items":
                    from agent.native_compaction import merge_interim_reasoning_items
                    last_msg[_key] = merge_interim_reasoning_items(last_msg.get(_key), interim_msg[_key])
                else:
                    last_msg[_key] = interim_msg[_key]
        else:
            append_message(messages, interim_msg)
            agent._emit_interim_assistant_message(interim_msg)

    if n < 3:
        # If the interim has nothing the Responses converter will replay, a bare retry is
        # byte-identical; a replayable interim holding only a ``compaction`` checkpoint
        # ALSO re-sends identically. One bare retry, then always nudge.
        interim_replayable = interim_has_content or interim_has_codex_reasoning or interim_has_codex_message_items
        if not interim_replayable or n >= 2:
            _last_msg = messages[-1] if messages else None
            if isinstance(_last_msg, dict):
                _already_nudged = (
                    _last_msg.get("role") == "user" and _last_msg.get("content") == _CODEX_INCOMPLETE_NUDGE
                )
                # Alternation guard: the nudge may only follow an assistant row.
                if not _already_nudged and _last_msg.get("role") == "assistant":
                    append_message(messages, {"role": "user", "content": _CODEX_INCOMPLETE_NUDGE})
        if not agent.quiet_mode:
            agent._vprint(f"{agent.log_prefix}↻ Codex response incomplete; continuing turn ({n}/3)")
        # Spinner/heartbeat notice: these retries can take minutes and otherwise look
        # like infinite thinking.
        # #70773: same FD-recycle corruption vector as #67142. The shared OpenAI client's connection pool
        # must NOT be closed from this watchdog/poll thread — worker threads from previous stale-killed
        # attempts may still be unwinding their SSL BIOs. The request-local client is already closed above
        # via _close_request_client_once. The shared client will be replaced lazily by
        # _ensure_primary_openai_client on the next request.
        # Surface the continuation on the live spinner/status line (CLI/TUI/Desktop) and gateway heartbeat:
        # each of these retries can spend minutes waiting on the provider, and without a distinct notice the
        # user only sees a generic thinking spinner ("infinite thinking", #64434).
        agent._emit_wait_notice(
            f"↻ model returned reasoning with no final answer — asking it to continue ({n}/3)"
        )
        agent._session_messages = messages
        return None

    agent._codex_incomplete_retries = 0
    agent._persist_session(messages, conversation_history)
    return partial_result(
        messages, api_call_count, "Codex response remained incomplete after 3 continuation attempts"
    )


@dataclass
class RefusalVerdict:
    """Outcome of ``handle_content_policy_refusal``: ``"break"`` (fallback activated —
    restart armed on ``_retry``; caller resets retry/compression counters) or
    ``"return"`` (the typed content-policy result in ``result``). ``active_system_prompt``
    is the possibly re-synced system prompt."""

    action: str
    result: Optional[Dict[str, Any]]
    active_system_prompt: Any


def handle_content_policy_refusal(
    agent: Any, response: Any, _retry: TurnRetryState, *, thinking_spinner: Any,
    messages: List[Dict[str, Any]], api_messages: Any, api_kwargs: Any, active_system_prompt: Any,
    conversation_history: Any, api_call_count: int, effective_task_id: Any, turn_id: Any,
    api_request_id: Any, api_start_time: float, retry_count: int, max_retries: int,
) -> RefusalVerdict:
    """HTTP-200 refusal (``finish_reason`` ``content_filter`` / ``guardrail_intervened``).
    Deterministic for the unchanged prompt — never retried: one configured-fallback try,
    else surface the refusal (explanation may live only in the reasoning channel)."""
    from agent.conversation_loop import (
        _CONTENT_POLICY_RECOVERY_HINT, _arm_fallback_restart, _content_policy_blocked_result
    )

    _refusal_result = normalize_response_for_agent(agent, response)
    _refusal_text = (getattr(_refusal_result, "content", None) or "").strip()
    if not _refusal_text:
        _refusal_text = (agent._extract_reasoning(_refusal_result) or "").strip()

    agent._invoke_api_request_error_hook(
        task_id=effective_task_id, turn_id=turn_id, api_request_id=api_request_id,
        api_call_count=api_call_count, api_start_time=api_start_time, api_kwargs=api_kwargs,
        error_type="ContentPolicyBlocked",
        error_message=_refusal_text or "model declined to respond (content_filter)",
        status_code=None, retry_count=retry_count, max_retries=max_retries, retryable=False,
        reason=FailoverReason.content_policy_blocked.value,
    )
    stop_thinking_spinner(agent, thinking_spinner)

    if agent._has_pending_fallback():
        agent._buffer_status("⚠️ Model declined to respond (safety refusal) — trying fallback...")
    if agent._try_activate_fallback():
        active_system_prompt = _arm_fallback_restart(agent, api_messages, active_system_prompt, _retry)
        return RefusalVerdict("break", None, active_system_prompt)

    agent._flush_status_buffer()
    _refusal_log = _refusal_text[:500] + "..." if len(_refusal_text) > 500 else _refusal_text
    logger.warning(
        "%sModel declined to respond (finish_reason=content_filter). model=%s provider=%s refusal=%s",
        agent.log_prefix, agent.model, agent.provider,
        _refusal_log or "(no text)",
    )
    agent._emit_status("⚠️ The model declined to respond to this request (safety refusal).")
    _refusal_detail = (
        f"Model's explanation: {_refusal_text}" if _refusal_text else "The model returned no explanation."
    )
    _refusal_response = (
        "⚠️  The model declined to respond to this request (safety refusal — not a Hermes/gateway failure).\n\n"
        f"{_refusal_detail}\n\n"
        f"{_CONTENT_POLICY_RECOVERY_HINT}"
    )
    agent._cleanup_task_resources(effective_task_id)
    agent._persist_session(messages, conversation_history)
    return RefusalVerdict("return", _content_policy_blocked_result(
        messages, api_call_count, final_response=_refusal_response,
        error_detail=_refusal_text or "model declined (content_filter)",
    ), active_system_prompt)
