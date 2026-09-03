"""WeCom TTFT progress hints: pushed while the stream bubble is empty, cancelled on real content."""
import asyncio
import importlib
import os
import tempfile

import pytest

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
os.environ["WECOM_ALLOW_ALL_USERS"] = "true"

m = importlib.import_module("plugins.platforms.wecom.adapter")
from gateway.config import PlatformConfig

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]


def _adapter(hints):
    ad = WeComAdapter(PlatformConfig(enabled=True, extra={"bot_id": "b", "secret": "s", "dm_policy": "open"}))
    ad._ttft_hints = hints
    sent = []

    async def fake_queued(req_id, body, is_final=False, skip_if_pending=False):
        sent.append(body["stream"]["content"])
        return {"errcode": 0}

    async def fake_reply(req_id, body, cmd="aibot_respond_msg", timeout=10.0):
        sent.append(body["stream"]["content"])
        return {"errcode": 0}

    ad._send_reply_queued = fake_queued
    ad._send_reply_request = fake_reply
    ad._last_chat_req_ids["c1"] = "r1"
    return ad, sent


def test_parse_hints():
    assert m._parse_ttft_hints("4:乙|2:甲|bad|:x|0:none") == [(2.0, "甲"), (4.0, "乙")]
    assert m._parse_ttft_hints("") == []
    assert m._parse_ttft_hints(m.TTFT_HINTS_DEFAULT)[0] == (2.0, "正在思考…")


def test_hints_fire_until_real_content():
    async def main():
        ad, sent = _adapter([(0.05, "甲"), (0.12, "乙"), (5.0, "丙")])
        assert await ad.send_stream_frame("", chat_id="c1", turn_id="t1")
        await asyncio.sleep(0.2)
        assert sent == ["<think></think>", "甲", "乙"], sent
        turn = ad._stream_turns["c1:t1"]
        assert turn.hint_task is not None
        assert await ad.send_stream_frame("真实内容", chat_id="c1", turn_id="t1")
        assert turn.real_content_sent and turn.hint_task is None
        await asyncio.sleep(0.05)
        assert sent[-1] == "真实内容" and "丙" not in sent
    asyncio.run(main())


def test_no_hint_when_content_arrives_first():
    async def main():
        ad, sent = _adapter([(0.05, "甲")])
        await ad.send_stream_frame("", chat_id="c1", turn_id="t1")
        await ad.send_stream_frame("快", chat_id="c1", turn_id="t1")
        await asyncio.sleep(0.1)
        assert "甲" not in sent, sent
    asyncio.run(main())


def test_finalize_cancels_hints():
    async def main():
        ad, sent = _adapter([(0.05, "甲"), (0.1, "乙")])
        await ad.send_stream_frame("", chat_id="c1", turn_id="t1")
        await asyncio.sleep(0.07)
        assert "甲" in sent
        await ad.send_stream_frame("完", chat_id="c1", turn_id="t1", finalize=True)
        await asyncio.sleep(0.1)
        assert "乙" not in sent, sent
        assert "c1:t1" not in ad._stream_turns
    asyncio.run(main())


def test_env_disables():
    assert m._parse_ttft_hints(None) == []
