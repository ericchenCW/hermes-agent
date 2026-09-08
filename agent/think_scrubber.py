"""Stateful scrubber for reasoning/thinking blocks in streamed assistant text.

The regex ``_strip_think_blocks`` is correct for a complete string but, run per-delta, erases an
opening ``<think>`` that arrives alone, so downstream state machines leak reasoning. This class
holds partial tags at delta boundaries until resolved; ``flush()`` releases held-back prose that
was not a tag; ``reset()`` at the top of each turn. An open tag only starts a block at a block
boundary (stream start / after a newline / whitespace-only line so far), so prose that *mentions*
``<think>`` is not suppressed; closed pairs are always suppressed (intentional).
"""

from __future__ import annotations

import os
import re
from typing import Tuple

__all__ = ["StreamingThinkScrubber"]

# Safety bound for an *unterminated* block (qwen3.6 gateway regression).
#
# When a provider emits a stray ``<think>`` marker on the content channel whose ``</think>`` never
# arrives (qwen3.6 streams its reasoning on the separate ``reasoning_content`` channel and only
# leaks the opening tag into ``content``), the scrubber used to stay ``_in_block`` for the rest of
# the stream and swallow the entire visible answer: the gateway saw thinking.delta frames and
# message.complete, but not a single message.delta.  Two guards bound that failure: ``flush()``
# surfaces the swallowed text when nothing else was ever emitted, and once more than this many
# characters have accumulated inside one unterminated block the open tag is retroactively treated
# as a misdetection so later deltas flow again.  A genuine reasoning block past 64k characters is
# not something the models on this path produce, whereas a 64k-character answer swallowed whole is
# a total loss.  Tune with ``HERMES_THINK_BLOCK_MAX_CHARS`` (<= 0 disables the guard).
_DEFAULT_MAX_BLOCK_CHARS = 64000


def _env_max_block_chars() -> int:
    raw = os.environ.get("HERMES_THINK_BLOCK_MAX_CHARS")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_BLOCK_CHARS
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return _DEFAULT_MAX_BLOCK_CHARS


class StreamingThinkScrubber:
    """Stateful scrubber for streaming reasoning/thinking blocks.

    State: ``_in_block`` (inside an open block; text discarded), ``_buf`` (held-back partial-tag
    tail), ``_last_emitted_ended_newline`` (True iff the last emission ended with ``\\n`` or nothing
    was emitted yet — decides whether an open tag at buffer position 0 sits at a block boundary).
    """

    _OPEN_TAG_NAMES: Tuple[str, ...] = ("think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD")

    # Lowercased literal tags so the hot path does string ops, not regex per feed().
    _OPEN_TAGS: Tuple[str, ...] = tuple(f"<{name.lower()}>" for name in _OPEN_TAG_NAMES)
    _CLOSE_TAGS: Tuple[str, ...] = tuple(f"</{name.lower()}>" for name in _OPEN_TAG_NAMES)
    _ALL_TAGS: Tuple[str, ...] = _OPEN_TAGS + _CLOSE_TAGS
    _MAX_TAG_LEN: int = max(len(tag) for tag in _ALL_TAGS)
    # Orphan close tag plus trailing whitespace (matches _strip_think_blocks case 3).
    _ORPHAN_CLOSE_RE = re.compile(
        "(?:" + "|".join(re.escape(t) for t in _CLOSE_TAGS) + r")[ \t\n\r]*", re.IGNORECASE
    )

    def __init__(self, max_block_chars: "int | None" = None) -> None:
        self._max_block_chars: int = (
            _env_max_block_chars() if max_block_chars is None else int(max_block_chars)
        )
        self.reset()

    def reset(self) -> None:
        """Reset all state.  Call at the top of every new turn."""
        self._in_block: bool = False
        self._buf: str = ""
        self._last_emitted_ended_newline: bool = True
        # Text discarded inside the *current* unterminated block, kept so flush() can surface it
        # when it turns out to have been the answer.
        self._block_text: str = ""
        # True once any visible character has been emitted in this stream.
        self._emitted_any_visible: bool = False

    def _emit(self, out: list[str], text: str) -> None:
        """Append visible prose to *out* (orphan close tags stripped) and track the newline flag."""
        text = self._strip_orphan_close_tags(text)
        if text:
            out.append(text)
            self._last_emitted_ended_newline = text.endswith("\n")
            self._emitted_any_visible = True

    def feed(self, text: str) -> str:
        """Feed one delta; return the scrubbed visible portion ("" when it is all reasoning or held back)."""
        if not text:
            return ""
        buf = self._buf + text
        self._buf = ""
        out: list[str] = []

        while buf:
            if self._in_block:
                close_idx, close_len = self._find_first_tag(buf, self._CLOSE_TAGS)
                if close_idx == -1:
                    # No close yet: hold back a possible partial close-tag prefix; the rest is
                    # dropped for now, but stashed in case flush() has to surface it.
                    self._block_text += self._hold_partial(buf, self._CLOSE_TAGS)
                    if self._block_overflowed():
                        # Retroactive misdetection: this "block" has eaten more than a plausible
                        # reasoning trace, so close it and let later deltas through again.
                        self._exit_block_on_overflow()
                    break
                buf = buf[close_idx + close_len:]
                self._in_block = False
                self._block_text = ""
                continue

            # Priority 1: closed <tag>X</tag> pair anywhere (even inline pairs are almost
            # certainly leaked reasoning). Priority 2: unterminated open tag at a block
            # boundary (gated so prose mentioning '<think>' isn't over-stripped). Earliest wins.
            pair = self._find_earliest_closed_pair(buf)
            open_idx, open_len = self._find_open_at_boundary(buf, out)
            if pair is not None and (open_idx == -1 or pair[0] <= open_idx):
                self._emit(out, buf[:pair[0]])
                buf = buf[pair[1]:]
                continue
            if open_idx != -1:
                self._emit(out, buf[:open_idx])
                self._in_block = True
                self._block_text = ""
                buf = buf[open_idx + open_len:]
                continue

            # No resolvable tag: hold back any partial-tag prefix at the tail
            # so a tag split across deltas isn't missed, then emit the rest.
            self._emit(out, self._hold_partial(buf, self._ALL_TAGS))
            break

        return "".join(out)

    def _hold_partial(self, buf: str, tags: Tuple[str, ...]) -> str:
        """Move a trailing partial-tag prefix of *buf* into ``_buf``; return the remainder."""
        held = self._max_partial_suffix(buf, tags)
        self._buf = buf[-held:] if held else ""
        return buf[:-held] if held else buf

    def flush(self) -> str:
        """End-of-stream flush.

        Inside an unterminated block the decision turns on whether the consumer already received
        anything visible.  Something emitted: the user has an answer, so the trailing unterminated
        block is almost certainly real reasoning cut short — drop it (leaking partial reasoning is
        worse than a truncated tail).  Nothing emitted at all: the alternative to surfacing the
        swallowed text is an empty reply, and a stream with zero visible characters is the signature
        of a stray, never-closed ``<think>`` on the content channel (qwen3.6 through the gateway), so
        the stashed text is emitted rather than lost.  The scrubber cannot tell reasoning from prose
        *inside* an unterminated block, so "did this stream produce any visible output at all" is the
        discriminator used instead.  Otherwise the tail is emitted verbatim.

        Always resets the boundary flag — intra-turn retries flush then stream again without
        ``reset()``, and a stale False flag made the new stream's opening ``<think>`` look mid-line.
        """
        rescued = self._block_text if (self._in_block and not self._emitted_any_visible) else ""
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._block_text = ""
        self._in_block = False
        self._last_emitted_ended_newline = True
        self._emitted_any_visible = False
        if rescued:
            return self._strip_orphan_close_tags(rescued).lstrip("\n")
        return self._strip_orphan_close_tags(tail) if tail else ""

    # ── internal helpers ───────────────────────────────────────────────

    def _block_overflowed(self) -> bool:
        return self._max_block_chars > 0 and len(self._block_text) > self._max_block_chars

    def _exit_block_on_overflow(self) -> None:
        """Treat the current unterminated block as a misdetection, releasing only *subsequent*
        deltas: emitting a 64k tail of possible reasoning is the worse failure.  Whatever follows
        starts a fresh line for boundary gating, so a leading '<think>' right after the release is
        still recognised as an opener."""
        self._in_block = False
        self._block_text = ""
        self._last_emitted_ended_newline = True

    @staticmethod
    def _find_first_tag(buf: str, tags: Tuple[str, ...]) -> Tuple[int, int]:
        """Return (earliest_index, tag_length) over *tags* (case-insensitive), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = [(idx, len(tag)) for tag in tags if (idx := buf_lower.find(tag)) != -1]
        return min(hits) if hits else (-1, 0)

    def _find_earliest_closed_pair(self, buf: str):
        """(start_idx, end_idx) of the earliest ``<tag>...</tag>`` pair (non-greedy, case-insensitive), else None."""
        buf_lower = buf.lower()
        pairs = []
        for open_tag, close_tag in zip(self._OPEN_TAGS, self._CLOSE_TAGS):
            open_idx = buf_lower.find(open_tag)
            close_idx = buf_lower.find(close_tag, open_idx + len(open_tag)) if open_idx != -1 else -1
            if close_idx != -1:
                pairs.append((open_idx, close_idx + len(close_tag)))
        return min(pairs) if pairs else None

    def _find_open_at_boundary(self, buf: str, already_emitted: list[str]) -> Tuple[int, int]:
        """Return the earliest block-boundary open-tag (idx, len), or (-1, 0)."""
        buf_lower = buf.lower()
        hits = []
        for tag in self._OPEN_TAGS:
            idx = buf_lower.find(tag)
            while idx != -1 and not self._is_block_boundary(buf, idx, already_emitted):
                idx = buf_lower.find(tag, idx + 1)
            if idx != -1:
                hits.append((idx, len(tag)))
        return min(hits) if hits else (-1, 0)

    def _is_block_boundary(self, buf: str, idx: int, already_emitted: list[str]) -> bool:
        """True iff *idx* is a block boundary: position 0 after a newline-terminated (or no) prior emission,
        or any position whose preceding text on the current line is whitespace-only (when no newline
        precedes it in *buf*, the prior emission must also have ended with a newline)."""
        prior_newline = already_emitted[-1].endswith("\n") if already_emitted else self._last_emitted_ended_newline
        if idx == 0:
            return prior_newline
        preceding = buf[:idx]
        last_nl = preceding.rfind("\n")
        return (prior_newline if last_nl == -1 else True) and preceding[last_nl + 1:].strip() == ""

    @classmethod
    def _max_partial_suffix(cls, buf: str, tags: Tuple[str, ...]) -> int:
        """Longest buf-suffix that is a strict prefix of any tag (full matches are real tags, handled elsewhere)."""
        buf_lower = buf.lower()
        for i in range(min(len(buf_lower), cls._MAX_TAG_LEN - 1), 0, -1):
            suffix = buf_lower[-i:]
            if any(len(tag) > i and tag.startswith(suffix) for tag in tags):
                return i
        return 0

    @classmethod
    def _strip_orphan_close_tags(cls, text: str) -> str:
        """Remove close tags with no matching open (always noise) plus trailing whitespace."""
        return cls._ORPHAN_CLOSE_RE.sub("", text) if "</" in text else text
