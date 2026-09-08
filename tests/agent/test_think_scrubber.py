"""Tests for StreamingThinkScrubber.

These tests lock in the contract the scrubber must satisfy so downstream
consumers (ACP, api_server, TTS, CLI, gateway) never see reasoning
blocks leaking through the stream_delta_callback.  The scenarios map
directly to the MiniMax-M2.7 / DeepSeek / Qwen3 streaming patterns that
break the older per-delta regex strip.
"""

from __future__ import annotations

import pytest

from agent.think_scrubber import StreamingThinkScrubber


def _drive(scrubber: StreamingThinkScrubber, deltas: list[str]) -> str:
    """Feed a sequence of deltas and return the concatenated visible output."""
    out = [scrubber.feed(d) for d in deltas]
    out.append(scrubber.flush())
    return "".join(out)


class TestClosedPairs:
    """Closed <tag>...</tag> pairs are always stripped, regardless of boundary."""

    def test_closed_pair_single_delta(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>reasoning</think>Hello world"]) == "Hello world"


    @pytest.mark.parametrize(
        "tag",
        ["think", "thinking", "reasoning", "thought", "REASONING_SCRATCHPAD"],
    )
    def test_all_tag_variants(self, tag: str) -> None:
        s = StreamingThinkScrubber()
        delta = f"<{tag}>x</{tag}>Hello"
        assert _drive(s, [delta]) == "Hello"



class TestUnterminatedOpen:
    """Unterminated open tag suppresses content until end of stream.

    At end of stream the swallowed text is rescued only when the stream
    produced nothing else — see ``TestUnterminatedBlockRescue``.
    """

    def test_open_at_stream_start(self) -> None:
        """Suppressed for the whole stream; surfaced by flush() because the
        alternative is an empty reply (qwen3.6 gateway regression)."""
        s = StreamingThinkScrubber()
        assert s.feed("<think>reasoning text with no close") == ""
        assert _drive(s, []) == "reasoning text with no close"



    def test_prose_mentioning_tag_not_stripped(self) -> None:
        """Mid-line '<think>' in prose is preserved (no boundary)."""
        s = StreamingThinkScrubber()
        text = "Use the <think> element for reasoning"
        assert _drive(s, [text]) == text


class TestOrphanClose:
    """Orphan close tags (no prior open) are stripped without boundary check."""

    def test_orphan_close_alone(self) -> None:
        s = StreamingThinkScrubber()
        assert _drive(s, ["Hello</think>world"]) == "Helloworld"




class TestPartialTagsAcrossDeltas:
    """Partial tags at delta boundaries must be held back, not emitted raw."""

    def test_split_open_tag_held_back(self) -> None:
        """'<' arrives alone, 'think>' completes it on next delta."""
        s = StreamingThinkScrubber()
        # At stream start, last_emitted_ended_newline=True, so <think> at 0 is boundary
        assert (
            _drive(s, ["<", "think>reasoning</think>done"])
            == "done"
        )

    def test_split_open_tag_not_at_boundary(self) -> None:
        """Mid-line split '<' + 'think>X</think>' is a closed pair.

        Closed pairs are always stripped (matching
        ``_strip_think_blocks`` case 1), even without a block
        boundary — a closed pair is an intentional bounded construct.
        """
        s = StreamingThinkScrubber()
        out = _drive(s, ["word<", "think>prose</think>more"])
        assert out == "wordmore"




class TestTheMiniMaxScenario:
    """The exact pattern run_agent per-delta regex strip breaks."""

    def test_minimax_split_open(self) -> None:
        """delta1='<think>', delta2='Let me check', delta3='</think>done'."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "Let me check their config", "</think>", "done"])
        assert out == "done"


    def test_minimax_unterminated_reasoning_at_end(self) -> None:
        """Unclosed reasoning at stream end is suppressed while streaming."""
        s = StreamingThinkScrubber()
        deltas = ["<think>", "The user wants", " to know something"]
        assert [s.feed(d) for d in deltas] == ["", "", ""]

    def test_unclosed_block_after_visible_text_is_dropped(self) -> None:
        """A trailing unterminated block is real reasoning: the user already
        has an answer, so flush() must not leak it."""
        s = StreamingThinkScrubber()
        out = _drive(s, ["Answer text.\n", "<think>", "second thoughts"])
        assert out == "Answer text.\n"


class TestUnterminatedBlockRescue:
    """qwen3.6 via the gateway: a stray ``<think>`` on the content channel
    whose ``</think>`` never arrives used to swallow the whole answer, so the
    gateway emitted thinking.delta + message.complete but zero message.delta.
    """

    def test_flush_rescues_answer_when_nothing_else_emitted(self) -> None:
        s = StreamingThinkScrubber()
        deltas = ["<think>", "The answer is 42.", "  Anything else?"]
        assert [s.feed(d) for d in deltas] == ["", "", ""]
        assert s.flush() == "The answer is 42.  Anything else?"
        # State is clean for the next stream.
        assert s._in_block is False
        assert s._block_text == ""
        assert _drive(s, ["<think>hidden</think>Hello"]) == "Hello"

    def test_rescue_only_once_not_replayed(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("<think>swallowed body")
        assert s.flush() == "swallowed body"
        assert s.flush() == ""

    def test_closed_block_still_fully_dropped(self) -> None:
        """A properly closed block must never be rescued, even when it is
        the only thing in the stream."""
        s = StreamingThinkScrubber()
        assert _drive(s, ["<think>secret</think>"]) == ""

    def test_overflow_releases_subsequent_deltas(self) -> None:
        """Once the "block" exceeds the bound it is treated as a
        misdetection and later deltas flow again."""
        s = StreamingThinkScrubber(max_block_chars=50)
        assert s.feed("<think>") == ""
        assert s.feed("x" * 51) == ""
        assert s._in_block is False
        assert s.feed("visible tail") == "visible tail"
        assert _drive(s, []) == ""

    def test_overflow_discards_the_swallowed_prefix(self) -> None:
        """The over-long prefix stays dropped: emitting a huge tail of
        possible reasoning is the worse failure mode."""
        s = StreamingThinkScrubber(max_block_chars=10)
        s.feed("<think>")
        s.feed("y" * 11)
        assert s.flush() == ""

    def test_overflow_bound_is_env_configurable(self, monkeypatch) -> None:
        monkeypatch.setenv("HERMES_THINK_BLOCK_MAX_CHARS", "5")
        s = StreamingThinkScrubber()
        assert s._max_block_chars == 5
        s.feed("<think>")
        s.feed("abcdef")
        assert s._in_block is False

    def test_overflow_guard_can_be_disabled(self, monkeypatch) -> None:
        monkeypatch.setenv("HERMES_THINK_BLOCK_MAX_CHARS", "0")
        s = StreamingThinkScrubber()
        s.feed("<think>")
        s.feed("z" * 100000)
        assert s._in_block is True

    def test_default_bound_does_not_trip_on_a_normal_block(self) -> None:
        s = StreamingThinkScrubber()
        out = _drive(s, ["<think>", "reasoning " * 500, "</think>", "Done."])
        assert out == "Done."


class TestResetAndReentry:
    def test_reset_clears_in_block_state(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("<think>hanging")
        assert s._in_block is True
        s.reset()
        assert s._in_block is False
        # After reset, a new turn works cleanly
        assert _drive(s, ["Hello world"]) == "Hello world"

    def test_reset_clears_buffered_partial_tag(self) -> None:
        s = StreamingThinkScrubber()
        s.feed("word<")
        assert s._buf == "<"
        s.reset()
        assert s._buf == ""
        assert _drive(s, ["fresh content"]) == "fresh content"


class TestFlushBehaviour:



    def test_flush_restores_stream_start_boundary(self) -> None:
        """End-of-stream flush must re-arm block-boundary gating.

        Thinking-only / empty-response retries flush then stream again
        without ``reset()``.  If flush left ``_last_emitted_ended_newline``
        False (e.g. after emitting a held-back ``<``), the next stream's
        opening ``<think>`` looked mid-line and leaked into the UI.
        """
        s = StreamingThinkScrubber()
        assert s.feed("word") == "word"
        assert s._last_emitted_ended_newline is False
        assert s.flush() == ""
        assert s._last_emitted_ended_newline is True
        assert (
            _drive(s, ["<think>", "secret reasoning", "</think>", "Visible answer"])
            == "Visible answer"
        )

    def test_flush_partial_tag_tail_does_not_poison_next_stream(self) -> None:
        """Flushing a held-back ``<`` must not make the next open tag leak."""
        s = StreamingThinkScrubber()
        s.feed("word<")
        assert s.flush() == "<"
        assert s._last_emitted_ended_newline is True
        assert _drive(s, ["<think>hidden</think>Hello"]) == "Hello"


class TestRealisticStreaming:
    """Character-by-character streaming must work as well as larger chunks."""

    def test_char_by_char_closed_pair(self) -> None:
        s = StreamingThinkScrubber()
        deltas = list("<think>x</think>Hello world")
        assert _drive(s, deltas) == "Hello world"


    def test_reasoning_then_real_response_first_word_preserved(self) -> None:
        """Regression: the first word of the final response must NOT be eaten.

        Stefan's screenshot bug — 'Let me check' was being rendered as
        ' me check'.  The scrubber must not consume any character of
        post-close content.
        """
        s = StreamingThinkScrubber()
        deltas = [
            "<think>",
            "User wants to know things",
            "</think>",
            "Let me check their config.",
        ]
        assert _drive(s, deltas) == "Let me check their config."

    def test_no_tag_passthrough_is_identical(self) -> None:
        """Streams without any reasoning tags pass through byte-for-byte."""
        s = StreamingThinkScrubber()
        deltas = ["Hello ", "world ", "how ", "are ", "you?"]
        assert _drive(s, deltas) == "Hello world how are you?"
