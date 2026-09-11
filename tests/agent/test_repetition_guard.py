"""Unit tests for the truncated-response repetition guard (issue #86581)."""

from __future__ import annotations

from agent.repetition_guard import MIN_FRAGMENT_LENGTH, is_repetition_dominated

# The exact sentence from the #86581 incident (echoed hundreds of times by
# the model before the provider cut it off at finish_reason=length).
_INCIDENT_ECHO = "好，你幫我更改成 Google Gemini 4 31B。"


class TestRepetitionGuard:
    def test_incident_shape_flags_repetition(self):
        # Narration + the echoed sentence on its own line, repeated (line path).
        text = ("We need to verify the model setting.\n" + _INCIDENT_ECHO + "\n") * 800
        assert is_repetition_dominated(text) is True

    def test_repeated_sentence_without_line_breaks_flags(self):
        # Repetition loop with no line breaks — exercises the window path.
        text = _INCIDENT_ECHO * 2000
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is True

    def test_long_legitimate_text_not_flagged(self):
        # Long, unique prose — no 60-char window ever repeats.
        text = " ".join(
            f"Sentence number {i} describes a distinct topic with unique words "
            f"such as quasar-{i} and nebula-{i} to keep every window distinct."
            for i in range(1200)
        )
        assert len(text) >= MIN_FRAGMENT_LENGTH
        assert is_repetition_dominated(text) is False

    def test_short_fragment_never_flagged(self):
        # Below MIN_FRAGMENT_LENGTH the guard fails open — short truncations
        # are legitimately continued even if they look repetitive.
        assert is_repetition_dominated("A. " * 50) is False
        assert is_repetition_dominated("hello ") is False

    def test_repeat_not_dominant_not_flagged(self):
        # A repeated sentence scattered through a long unique text: repeated
        # windows exist but cover far less than half of the fragment.
        filler = " ".join(f"unique filler token {i}" for i in range(3000))
        text = filler + ("\n" + _INCIDENT_ECHO + "\n") * 30
        assert is_repetition_dominated(text) is False

    def test_non_string_inputs(self):
        assert is_repetition_dominated("") is False
        assert is_repetition_dominated(None) is False
        assert is_repetition_dominated(12345) is False


# ── streaming verbatim-tail probe: the 200-character floor ─────────────
# 2026-09-11 regression, item 12: 「写一段 50 字的英文自我介绍」 was aborted by the
# verbatim tail probe.  The floor that already governs the normalized-line probe
# now governs this one too — and, critically, it is measured on the WHOLE reply,
# not on the pre-sliced trailing window the streaming caller hands over (that
# slice reaches 2048 characters long before the floor could ever bind).

from agent import repetition_guard as rg  # noqa: E402

# A 50-word English self-introduction of the kind the model was asked for.
# It legitimately repeats a few phrases ("I help", "I can") — ordinary prose.
_SHORT_SELF_INTRO = (
    "Hi, I am an IT support assistant. I help colleagues with VPN access, "
    "mailbox setup and printer问题 every day. I can look things up in the "
    "knowledge base, I can walk you through the steps, and I can escalate to "
    "a human on duty when I cannot solve it myself."
)

# The 2026-09-09 CaoDi-style degeneration: one paragraph, verbatim, over and
# over inside the reasoning stream.
_DEGENERATE_PARAGRAPH = (
    "现在让我检查第四步的表格内容。我发现表格中只有广州、深圳、北京三个城市的信息，"
    "缺少上海这一行。让我构造一个精确的替换操作，把上海这一行加进去。"
)


class TestTailProbeFloor:
    def test_a_short_english_self_introduction_never_trips(self):
        assert len(_SHORT_SELF_INTRO) < rg.STREAM_MIN_REPLY_CHARS * 2
        assert rg.tail_repetition_detected(_SHORT_SELF_INTRO) is False

    def test_nothing_under_the_floor_is_ever_judged(self):
        # Even a blatant verbatim loop stays unjudged while the reply is short.
        loop = "abcdefghij" * 15  # 150 chars
        assert len(loop) < rg.STREAM_MIN_REPLY_CHARS
        assert rg.tail_repetition_detected(loop, fragment=10, min_repeats=3) is False

    def test_the_floor_is_measured_on_the_whole_reply_not_the_slice(self):
        """The streaming caller passes only the tail; ``total_chars`` carries the floor."""
        tail = _DEGENERATE_PARAGRAPH * 12
        assert rg.tail_repetition_detected(tail, total_chars=len(tail)) is True
        assert rg.tail_repetition_detected(tail, total_chars=199) is False
        assert rg.tail_repetition_detected(tail, total_chars=rg.STREAM_MIN_REPLY_CHARS) is True

    def test_the_degenerate_paragraph_still_trips_well_under_4000_chars(self):
        text = _DEGENERATE_PARAGRAPH * 12
        assert len(text) <= 4000
        assert rg.tail_repetition_detected(text) is True
