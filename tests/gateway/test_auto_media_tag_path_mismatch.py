"""Auto-appended MEDIA tags must not depend on the model retyping the path correctly.

Production shape (spark, 2026-09-15 09:35): ``compose_sheet`` produced
``/opt/data/cache/sheets/steps-1789464927863.jpg``; the model's final reply carried
``MEDIA:/opt/data/cache/sheets/st-1789464927863.jpg``. ``_append_auto_media_tags`` bailed out on the
bare substring ``"MEDIA:" in final_response``, so the real tag was never appended, and delivery then
logged ``Skipping MEDIA directive path (not found on this host)`` — the long image the tool had
already built was never sent. At 09:37:50 the same code path delivered fine (``upload_media_bytes
OK``): the only difference was that the model had copied the filename correctly.
"""

from gateway.run_turn_runner import TurnRunner

REAL = "/opt/data/cache/sheets/steps-1789464927863.jpg"
TYPO = "/opt/data/cache/sheets/st-1789464927863.jpg"


def _result(path=REAL):
    return {
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "compose_sheet"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": f"图 1: a\nMEDIA:{path}"},
        ]
    }


def _append(final_response, result=None):
    return TurnRunner._append_auto_media_tags(
        None, final_response, result or _result(), [], set())


def test_mistyped_media_path_still_gets_the_real_tag_appended():
    out = _append(f"广州打印机扫描步骤见下图。\nMEDIA:{TYPO}")
    assert f"MEDIA:{REAL}" in out, out


def test_correctly_transcribed_path_is_not_appended_twice():
    reply = f"步骤见下图。\nMEDIA:{REAL}"
    assert _append(reply) == reply


def test_reply_without_any_media_line_still_gets_the_tag():
    out = _append("步骤见下图。")
    assert out.endswith(f"MEDIA:{REAL}")


def test_no_producer_output_leaves_the_reply_alone():
    reply = f"文档里写着 MEDIA:{TYPO} 这种示例"
    assert _append(reply, result={"messages": []}) == reply
