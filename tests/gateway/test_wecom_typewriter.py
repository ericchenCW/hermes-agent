"""WeCom typewriter pacer: cumulative text is revealed in small steps; finalize flushes; hints are random pools."""
import asyncio
import importlib
import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
os.environ["WECOM_ALLOW_ALL_USERS"] = "true"

m = importlib.import_module("plugins.platforms.wecom.adapter")
from gateway.config import PlatformConfig

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]


def _adapter():
    ad = WeComAdapter(PlatformConfig(enabled=True, extra={"bot_id": "b", "secret": "s", "dm_policy": "open"}))
    ad._ttft_hints = []
    sent = []

    async def fake_queued(req_id, body, is_final=False, skip_if_pending=False):
        sent.append((body["stream"]["content"], bool(body["stream"]["finish"])))
        return {"errcode": 0}

    async def fake_reply(req_id, body, cmd="aibot_respond_msg", timeout=10.0):
        sent.append((body["stream"]["content"], bool(body["stream"]["finish"])))
        return {"errcode": 0}

    ad._send_reply_queued = fake_queued
    ad._send_reply_request = fake_reply
    ad._last_chat_req_ids["c1"] = "r1"
    return ad, sent


def test_parse_pools():
    parsed = m._parse_ttft_hints("4:乙/丁|2:甲/丙")
    assert parsed == [(2.0, ["甲", "丙"]), (4.0, ["乙", "丁"])]
    assert all(len(pool) >= 3 for _, pool in m._parse_ttft_hints(m.TTFT_HINTS_DEFAULT))


def test_typewriter_next_steps():
    assert m.TYPEWRITER_MIN_STEP <= len(WeComAdapter._typewriter_next("", "一" * 100)) <= m.TYPEWRITER_MAX_STEP
    assert WeComAdapter._typewriter_next("ab", "abcd") == "abcd"
    assert WeComAdapter._typewriter_next("xyz", "abcdefgh") == "abcdefgh"[: m.TYPEWRITER_MIN_STEP]  # rewrite restarts


def test_pacer_reveals_gradually_then_finalize(monkeypatch):
    monkeypatch.setattr(m, "TYPEWRITER_TICK_SECONDS", 0.02)
    async def main():
        ad, sent = _adapter()
        assert await ad.send_stream_frame("", chat_id="c1", turn_id="t1")
        text = "".join("第%d句。" % i for i in range(30))  # 120 chars
        assert await ad.send_stream_frame(text, chat_id="c1", turn_id="t1")
        turn = ad._stream_turns["c1:t1"]
        assert turn.pacer_task is not None
        await asyncio.sleep(0.4)
        frames = [c for c, fin in sent if not fin and c != "<think></think>"]
        assert len(frames) >= 4, frames
        assert all(text.startswith(f) for f in frames)
        assert all(len(frames[i]) < len(frames[i + 1]) for i in range(len(frames) - 1))
        assert frames[-1] == text
        assert await ad.send_stream_frame(text + "完", chat_id="c1", turn_id="t1", finalize=True)
        assert sent[-1] == (text + "完", True)
        assert "c1:t1" not in ad._stream_turns
    asyncio.run(main())


def test_hints_random_without_immediate_repeat():
    async def main():
        ad, sent = _adapter()
        ad._ttft_hints = [(0.03, ["甲", "乙"]), (0.06, ["甲", "乙"]), (0.09, ["甲", "乙"])]
        await ad.send_stream_frame("", chat_id="c1", turn_id="t1")
        await asyncio.sleep(0.2)
        hints = [c for c, fin in sent if c != "<think></think>"]
        assert len(hints) == 3 and all(hints[i] != hints[i + 1] for i in range(2)), hints
    asyncio.run(main())


def test_pacer_respects_frame_budget(monkeypatch):
    monkeypatch.setattr(m, "TYPEWRITER_TICK_SECONDS", 0.01)
    async def main():
        ad, sent = _adapter()
        await ad.send_stream_frame("", chat_id="c1", turn_id="t1")
        turn = ad._stream_turns["c1:t1"]
        turn._intermediate_frames_sent = m.MAX_INTERMEDIATE_FRAMES - 1
        await ad.send_stream_frame("一" * 500, chat_id="c1", turn_id="t1")
        await asyncio.sleep(0.15)
        frames = [c for c, fin in sent if not fin and c != "<think></think>"]
        assert len(frames) == 1, len(frames)
        await ad.send_stream_frame("一" * 500, chat_id="c1", turn_id="t1", finalize=True)
        assert sent[-1][1] is True and len(sent[-1][0]) >= 500
    asyncio.run(main())
