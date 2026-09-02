"""Button cards (idcsre patch): ``BUTTONS[title]: a | b`` directive → WeCom
button_interaction template card, and template_card_event → user message."""
import asyncio
import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
os.environ["WECOM_ALLOW_ALL_USERS"] = "true"

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.wecom.adapter as m

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]


def _adapter():
    return WeComAdapter(PlatformConfig(enabled=True, extra={"bot_id": "b", "secret": "s", "dm_policy": "open"}))


def _wire(ad):
    sent = []

    async def fake_reply(req_id, body, cmd="aibot_respond_msg", timeout=10.0):
        sent.append(("reply", req_id, cmd, body)); return {"errcode": 0}

    async def fake_queued(req_id, body, is_final=False, skip_if_pending=False):
        sent.append(("queued", req_id, body)); return {"errcode": 0}

    async def fake_send(cmd, body, timeout=10.0):
        sent.append(("send", cmd, body)); return {"errcode": 0}

    ad._send_reply_request = fake_reply
    ad._send_reply_queued = fake_queued
    ad._send_request = fake_send
    return sent


def test_directive_parsing_and_card_shape():
    ad = _adapter()
    clean, spec = ad._extract_button_directive("先选一下城市：\nBUTTONS[请选择所在城市]: 广州 | 深圳 ｜ 上海 | 北京 | 广州\n")
    assert clean == "先选一下城市："
    assert spec == {"title": "请选择所在城市", "options": ["广州", "深圳", "上海", "北京"]}
    _, spec2 = ad._extract_button_directive("**BUTTONS**: Windows | Mac")
    assert spec2 == {"title": m.BUTTON_DEFAULT_TITLE, "options": ["Windows", "Mac"]}
    assert ad._extract_button_directive("正文里提到 buttons 但没有指令") == ("正文里提到 buttons 但没有指令", None)
    _, spec3 = ad._extract_button_directive("BUTTONS[x]: " + " | ".join(f"o{i}" for i in range(9)))
    assert len(spec3["options"]) == m.BUTTON_MAX
    task_id, card = ad._build_button_card("chat1", spec)
    assert card["card_type"] == "button_interaction" and card["task_id"] == task_id
    assert [b["text"] for b in card["button_list"]] == ["广州", "深圳", "上海", "北京"]
    assert [b["key"] for b in card["button_list"]] == ["opt0", "opt1", "opt2", "opt3"]
    assert ad._pending_button_cards[task_id]["chat_id"] == "chat1"


def test_click_updates_card_and_routes_label():
    ad = _adapter()
    sent = _wire(ad)
    routed = []

    async def fake_handle(event):
        routed.append(event)

    ad.handle_message = fake_handle

    async def run():
        assert await ad._send_button_card("user1", {"title": "t", "options": ["广州", "深圳", "上海"]}, None) is True
        assert sent[-1][0] == "send" and sent[-1][2]["msgtype"] == "template_card"
        tid = sent[-1][2]["template_card"]["task_id"]
        payload = {"cmd": m.APP_CMD_EVENT_CALLBACK, "headers": {"req_id": "evt-9"},
                   "body": {"msgid": "m-1", "chattype": "single", "from": {"userid": "user1"}, "msgtype": "event",
                            "event": {"eventtype": "template_card_event",
                                      "template_card_event": {"card_type": "button_interaction", "event_key": "opt2", "task_id": tid}}}}
        await ad._dispatch_payload(payload)
        upd = [x for x in sent if x[0] == "reply" and x[2] == m.APP_CMD_RESPONSE_UPDATE]
        assert upd and upd[-1][1] == "evt-9"
        assert upd[-1][3]["response_type"] == "update_template_card"
        assert upd[-1][3]["template_card"]["task_id"] == tid
        assert upd[-1][3]["template_card"]["sub_title_text"] == "已选择：上海"
        assert routed and routed[-1].text == "上海" and routed[-1].source.chat_id == "user1"
        assert ad._last_chat_req_ids.get("user1") == "evt-9"
        assert tid not in ad._pending_button_cards
        # unknown task id → key used as label, still routed
        payload["body"]["msgid"] = "m-2"
        payload["body"]["event"]["template_card_event"] = {"event_key": "yes", "task_id": "nope"}
        await ad._dispatch_payload(payload)
        assert routed[-1].text == "yes"
        # feedback events are logged, not routed
        payload["body"]["msgid"] = "m-3"
        payload["body"]["event"] = {"eventtype": "feedback_event", "feedback_event": {"type": 1}}
        n = len(routed)
        await ad._dispatch_payload(payload)
        assert len(routed) == n

    asyncio.run(run())


def test_send_embeds_card_in_finish_frame_or_sends_proactively():
    ad = _adapter()
    sent = _wire(ad)

    async def run():
        ad._last_chat_req_ids["user1"] = "req-1"
        r = await ad.send("user1", "先选城市：\nBUTTONS[请选择城市]: 广州 | 深圳")
        assert r.success
        assert len(sent) == 1 and sent[0][0] == "queued"
        b = sent[0][2]
        assert b["msgtype"] == "stream_with_template_card" and b["stream"]["finish"] is True
        assert b["stream"]["content"] == "先选城市："
        assert [x["text"] for x in b["template_card"]["button_list"]] == ["广州", "深圳"]
        sent.clear()
        assert (await ad.send("user1", "普通回复")).success
        assert len(sent) == 1 and sent[0][-1]["msgtype"] == "markdown"
        sent.clear()
        assert (await ad.send("user2", "系统？\nBUTTONS: Windows | Mac")).success
        assert [x[0] for x in sent] == ["send", "send"]
        assert sent[0][2]["msgtype"] == "markdown" and sent[0][2]["markdown"]["content"] == "系统？"
        assert sent[1][2]["msgtype"] == "template_card" and sent[1][2]["chatid"] == "user2"
        sent.clear()
        await ad._send_stream_reply("rq", "sid", "hi", finish=True, template_card={"card_type": "button_interaction", "task_id": "t"})
        assert sent[-1][2]["msgtype"] == "stream_with_template_card"
        await ad._send_stream_reply("rq", "sid", "hi", finish=False, template_card={"card_type": "x"})
        assert sent[-1][2]["msgtype"] == "stream" and "template_card" not in sent[-1][2]

    asyncio.run(run())
