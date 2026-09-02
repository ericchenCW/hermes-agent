"""Button cards (idcsre patch): ``BUTTONS[title]: a | b`` directive → WeCom
button_interaction template card, and template_card_event → user message."""
import asyncio
import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
os.environ["WECOM_ALLOW_ALL_USERS"] = "true"

from gateway.config import PlatformConfig
import plugins.platforms.wecom.adapter as m

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]


def _adapter():
    return WeComAdapter(PlatformConfig(enabled=True, extra={"bot_id": "b", "secret": "s", "dm_policy": "open", "group_policy": "open"}))


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


def _click(task_id, key, *, msgid, req_id="evt-9", chattype="single", userid="user1", chatid=None, flat=False):
    detail = {"card_type": "button_interaction", "event_key": key, "task_id": task_id}
    event = {"eventtype": "template_card_event", **detail} if flat else {"eventtype": "template_card_event", "template_card_event": detail}
    body = {"msgid": msgid, "chattype": chattype, "from": {"userid": userid}, "msgtype": "event", "event": event}
    if chatid:
        body["chatid"] = chatid
    return {"cmd": m.APP_CMD_EVENT_CALLBACK, "headers": {"req_id": req_id}, "body": body}


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


def test_directive_parsing_and_card_shape():
    ad = _adapter()
    clean, spec = ad._extract_button_directive("先选一下城市：\nBUTTONS[请选择所在城市]: 广州 | 深圳 ｜ 上海 | 北京 | 广州\n")
    assert clean == "先选一下城市："
    assert spec == {"title": "请选择所在城市", "options": ["广州", "深圳", "上海", "北京"]}
    _, spec2 = ad._extract_button_directive("**BUTTONS**: Windows | Mac")
    assert spec2 == {"title": m.BUTTON_DEFAULT_TITLE, "options": ["Windows", "Mac"]}
    assert ad._extract_button_directive("正文里提到 buttons 但没有指令") == ("正文里提到 buttons 但没有指令", None)
    # only the last line counts; a directive quoted mid-text stays put
    mid = "用法：\nBUTTONS[x]: a | b\n以上就是用法。"
    assert ad._extract_button_directive(mid) == (mid, None)
    # inside an open code fence → not a directive
    fenced = "文档里写：\n```\nBUTTONS[请选择所在城市]: 广州 | 深圳\n```"
    assert ad._extract_button_directive(fenced) == (fenced, None)
    _, spec3 = ad._extract_button_directive("BUTTONS[x]: " + " | ".join(f"o{i}" for i in range(9)))
    assert len(spec3["options"]) == m.BUTTON_MAX
    _, spec4 = ad._extract_button_directive("BUTTONS: 广州天河办公室打印机 | 广州天河办公室扫描仪")
    assert len(spec4["options"]) == 2 and all(len(o) <= m.BUTTON_LABEL_MAX for o in spec4["options"])
    task_id, card = ad._build_button_card("chat1", spec)
    assert card["card_type"] == "button_interaction" and card["task_id"] == task_id
    assert [b["text"] for b in card["button_list"]] == ["广州", "深圳", "上海", "北京"]
    assert [b["key"] for b in card["button_list"]] == ["opt0|广州", "opt1|深圳", "opt2|上海", "opt3|北京"]
    assert ad._pending_button_cards[task_id]["chat_id"] == "chat1"
    # partial directive hidden from intermediate frames
    assert ad._strip_partial_button_line("正文\nBUTTONS[请选") == "正文"
    assert ad._strip_partial_button_line("正文\n**BUTTONS") == "正文"
    assert ad._strip_partial_button_line("正文没有指令") == "正文没有指令"


def test_click_updates_card_and_routes_label():
    ad = _adapter()
    sent = _wire(ad)
    routed = []

    async def fake_handle(event):
        routed.append(event)

    ad.handle_message = fake_handle

    async def run():
        ad._last_chat_req_ids["user1"] = "old-req"
        assert await ad._send_button_card("user1", {"title": "t", "options": ["广州", "深圳", "上海"]}, None) is True
        assert sent[-1][0] == "send" and sent[-1][2]["msgtype"] == "template_card"
        tid = sent[-1][2]["template_card"]["task_id"]
        await ad._dispatch_payload(_click(tid, "opt2|上海", msgid="m-1"))
        await _settle()
        upd = [x for x in sent if x[0] == "reply" and x[2] == m.APP_CMD_RESPONSE_UPDATE]
        assert upd and upd[-1][1] == "evt-9"
        assert upd[-1][3]["response_type"] == "update_template_card"
        assert upd[-1][3]["template_card"]["task_id"] == tid
        assert upd[-1][3]["template_card"]["sub_title_text"] == "已选择：上海"
        assert routed and routed[-1].text == "上海" and routed[-1].source.chat_id == "user1"
        # the event req_id must NOT become the chat's reply req_id; stale one dropped
        assert "user1" not in ad._last_chat_req_ids
        assert "user1" in ad._stream_expired_chats and "user1" in ad._button_click_chats
        assert ad._pending_button_cards[tid]["consumed"] is True
        # repeat click → update frame only, nothing routed
        n_upd, n_routed = len(upd), len(routed)
        await ad._dispatch_payload(_click(tid, "opt1|深圳", msgid="m-1b", req_id="evt-10"))
        await _settle()
        assert len([x for x in sent if x[0] == "reply" and x[2] == m.APP_CMD_RESPONSE_UPDATE]) == n_upd + 1
        assert len(routed) == n_routed
        # unknown task id (registry lost) → label decoded from the key; flat event shape
        await ad._dispatch_payload(_click("nope", "opt1|深圳", msgid="m-2", flat=True))
        await _settle()
        assert routed[-1].text == "深圳"
        # group click: chat resolved from the card registry even without body.chatid
        ad._group_chat_ids.add("grp-1")
        assert await ad._send_button_card("grp-1", {"title": "t", "options": ["A", "B"]}, None) is True
        gtid = sent[-1][2]["template_card"]["task_id"]
        await ad._dispatch_payload(_click(gtid, "opt0|A", msgid="m-3", userid="user3"))
        await _settle()
        assert routed[-1].text == "A" and routed[-1].source.chat_id == "grp-1" and routed[-1].source.chat_type == "group"
        # feedback events are logged, not routed
        n = len(routed)
        p = _click("x", "y", msgid="m-4")
        p["body"]["event"] = {"eventtype": "feedback_event", "feedback_event": {"type": 1}}
        await ad._dispatch_payload(p)
        await _settle()
        assert len(routed) == n
        # a real inbound message clears the click marker
        ad._remember_chat_req_id("user1", "req-new")
        assert "user1" not in ad._button_click_chats and "user1" not in ad._stream_expired_chats

    asyncio.run(run())


def test_click_handler_runs_off_the_read_loop():
    ad = _adapter()

    async def run():
        gate = asyncio.Event()

        async def slow_reply(req_id, body, cmd="aibot_respond_msg", timeout=10.0):
            await gate.wait(); return {"errcode": 0}

        ad._send_reply_request = slow_reply
        _, card = ad._build_button_card("user1", {"title": "t", "options": ["A"]})
        t0 = asyncio.get_event_loop().time()
        await ad._dispatch_payload(_click(card["task_id"], "opt0|A", msgid="m-1"))
        # dispatch returned without waiting for the ack
        assert asyncio.get_event_loop().time() - t0 < 0.5
        gate.set()
        await _settle()

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
        # no req_id (DM): markdown then card, both proactive
        sent.clear()
        assert (await ad.send("user2", "系统？\nBUTTONS: Windows | Mac")).success
        assert [x[0] for x in sent] == ["send", "send"]
        assert sent[0][2]["msgtype"] == "markdown" and sent[0][2]["markdown"]["content"] == "系统？"
        assert sent[1][2]["msgtype"] == "template_card" and sent[1][2]["chatid"] == "user2"
        # group without req_id: refused normally, proactive attempt on a button-click turn
        sent.clear()
        ad._group_chat_ids.add("grp-1")
        assert not (await ad.send("grp-1", "hi")).success and not sent
        ad._button_click_chats.add("grp-1")
        assert (await ad.send("grp-1", "hi")).success and sent[0][0] == "send" and sent[0][2]["chatid"] == "grp-1"
        # directive only → title used as the text
        sent.clear()
        assert (await ad.send("user2", "BUTTONS[要继续吗]: 是 | 否")).success
        assert sent[0][2]["markdown"]["content"] == "要继续吗"
        # _send_stream_reply: finish frame carries the card, intermediate never does
        sent.clear()
        await ad._send_stream_reply("rq", "sid", "hi", finish=True, template_card={"card_type": "button_interaction", "task_id": "t"})
        assert sent[-1][2]["msgtype"] == "stream_with_template_card"
        await ad._send_stream_reply("rq", "sid", "hi", finish=False, template_card={"card_type": "x"})
        assert sent[-1][2]["msgtype"] == "stream" and "template_card" not in sent[-1][2]

    asyncio.run(run())
