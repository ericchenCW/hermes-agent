"""Explicit WeCom cards: ``adapter.send_card(chat_id, card, fallback_text)``.

The ``platform_send`` card path hands a structured spec straight to the adapter —
nothing is parsed out of reply text, and the caller's button keys ride through
untouched. A delivery failure degrades to the fallback text so the notification
is never silently lost.
"""
import asyncio
import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
os.environ["WECOM_ALLOW_ALL_USERS"] = "true"

from gateway.config import PlatformConfig
import plugins.platforms.wecom.adapter as m
import plugins.platforms.wecom.buttons as mb

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]

KEY_APPROVE = "iac:ap-42:9f3c7d1e:approve"
KEY_REJECT = "iac:ap-42:9f3c7d1e:reject"

CARD = {
    "title": "IaC 变更审批",
    "desc": "变更 3 台主机的安全组",
    "buttons": [{"key": KEY_APPROVE, "text": "批准", "style": 1},
                {"key": KEY_REJECT, "text": "拒绝", "style": 2}],
    "url": "https://haro.example/iac/ap-42",
}


def _adapter():
    return WeComAdapter(PlatformConfig(enabled=True, extra={
        "bot_id": "b", "secret": "s", "dm_policy": "open", "group_policy": "open"}))


def _wire(ad, *, send_errcode=0):
    sent = []

    async def fake_send(cmd, body, timeout=10.0):
        sent.append(("send", cmd, body))
        # only the card frame fails: the plain-text fallback must still get through
        failing = send_errcode if body.get("msgtype") == "template_card" else 0
        return {"errcode": failing, "errmsg": "boom" if failing else "ok"}

    async def fake_reply(req_id, body, cmd="aibot_respond_msg", timeout=10.0):
        sent.append(("reply", req_id, cmd, body))
        return {"errcode": 0}

    ad._send_request = fake_send
    ad._send_reply_request = fake_reply
    return sent


def test_send_card_builds_a_button_interaction_card():
    ad = _adapter()
    sent = _wire(ad)
    result = asyncio.run(ad.send_card("ericyu", CARD, "兜底文本"))

    assert result.success
    kind, cmd, body = sent[0]
    assert (kind, cmd) == ("send", m.APP_CMD_SEND)
    assert body["chatid"] == "ericyu" and body["msgtype"] == "template_card"
    card = body["template_card"]
    assert card["card_type"] == "button_interaction"
    assert card["main_title"] == {"title": "IaC 变更审批", "desc": "变更 3 台主机的安全组"}
    assert card["button_list"] == [
        {"text": "批准", "style": 1, "key": KEY_APPROVE},
        {"text": "拒绝", "style": 2, "key": KEY_REJECT},
    ]
    assert card["task_id"].startswith(mb.CARD_TASK_ID_PREFIX)


def test_send_card_registers_title_and_url_for_the_click_ack():
    ad = _adapter()
    sent = _wire(ad)
    asyncio.run(ad.send_card("ericyu", CARD, "兜底文本"))
    task_id = sent[0][2]["template_card"]["task_id"]
    pending = ad._pending_button_cards[task_id]
    assert pending["chat_id"] == "ericyu"
    assert pending["title"] == "IaC 变更审批"
    assert pending["url"] == "https://haro.example/iac/ap-42"
    assert pending["explicit"] is True


def test_send_card_truncates_to_the_wecom_limits():
    ad = _adapter()
    sent = _wire(ad)
    asyncio.run(ad.send_card("ericyu", {
        "title": "标" * 40, "desc": "描" * 200,
        "buttons": [{"key": "k", "text": "文" * 40}],
    }, ""))
    card = sent[0][2]["template_card"]
    assert len(card["main_title"]["title"]) == mb.BUTTON_TITLE_MAX
    assert len(card["main_title"]["desc"]) == mb.CARD_DESC_MAX
    assert len(card["button_list"][0]["text"]) == mb.BUTTON_LABEL_MAX
    assert card["button_list"][0]["key"] == "k"  # keys are never truncated


def test_card_failure_falls_back_to_the_text_body():
    ad = _adapter()
    sent = _wire(ad, send_errcode=42045)
    result = asyncio.run(ad.send_card("ericyu", CARD, "兜底文本"))
    assert result.success  # the fallback markdown send succeeded
    kinds = [(entry[1], entry[2].get("msgtype")) for entry in sent if entry[0] == "send"]
    assert kinds[0] == (m.APP_CMD_SEND, "template_card")
    assert kinds[1] == (m.APP_CMD_SEND, "markdown")
    assert sent[-1][2]["markdown"]["content"] == "兜底文本"
    # the aborted card left nothing in the registry
    assert ad._pending_button_cards == {}


def test_card_failure_without_a_fallback_reports_the_failure():
    ad = _adapter()
    _wire(ad, send_errcode=42045)
    result = asyncio.run(ad.send_card("ericyu", CARD, ""))
    assert not result.success and "card delivery failed" in result.error


def test_send_card_requires_a_chat_id():
    ad = _adapter()
    _wire(ad)
    assert not asyncio.run(ad.send_card("", CARD, "x")).success


def test_platform_send_text_path_does_not_parse_a_buttons_directive():
    ad = _adapter()
    sent = _wire(ad)
    body = "请选择：\nBUTTONS[请选择]: 批准 | 拒绝"
    result = asyncio.run(ad.send("ericyu", body, metadata={"no_button_directive": True}))
    assert result.success
    payloads = [entry[2] for entry in sent if entry[0] == "send"]
    assert len(payloads) == 1 and payloads[0]["msgtype"] == "markdown"
    assert payloads[0]["markdown"]["content"] == body  # verbatim, directive included
    assert ad._pending_button_cards == {}
