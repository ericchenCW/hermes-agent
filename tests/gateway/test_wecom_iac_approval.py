"""IaC approval callback: ``iac:<approvalId>:<nonce>:<decision>`` button clicks.

Haro owns the decision (nonce, binding, permissions); the adapter only routes the
click and rewrites the card inside WeCom's 5 s ack window. Group clicks are
refused locally, and the nonce never reaches a log line or the user.
"""
import asyncio
import logging
import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())
os.environ["WECOM_ALLOW_ALL_USERS"] = "true"
os.environ["WECOM_BUTTONS"] = "1"

import pytest

from gateway.config import PlatformConfig
import plugins.platforms.wecom.adapter as m
import plugins.platforms.wecom.iac_approval as iac

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]

NONCE = "9f3c7d1e5b"
KEY_APPROVE = f"iac:ap-42:{NONCE}:approve"
KEY_REJECT = f"iac:ap-42:{NONCE}:reject"


@pytest.fixture(autouse=True)
def _haro_env(monkeypatch):
    monkeypatch.setenv(iac.IAC_API_URL_ENV, "http://haro.internal:8000/")
    monkeypatch.setenv(iac.IAC_TOKEN_ENV, "rt-secret-token")


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeClient:
    """httpx.AsyncClient stand-in recording the single decide call."""

    def __init__(self, response=None, raises=None, delay=0.0):
        self.calls = []
        self._response = response
        self._raises = raises
        self._delay = delay

    async def post(self, url, json=None, headers=None, **kwargs):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        return self._response


def _adapter(client=None):
    ad = WeComAdapter(PlatformConfig(enabled=True, extra={
        "bot_id": "b", "secret": "s", "dm_policy": "open", "group_policy": "open"}))
    ad._iac_http_client = client
    return ad


def _wire(ad):
    """Capture card updates; the routing path must never be reached for iac keys."""
    updates = []
    routed = []

    async def fake_reply(req_id, body, cmd="aibot_respond_msg", timeout=10.0):
        updates.append((req_id, cmd, body))
        return {"errcode": 0}

    async def fake_handle(event):
        routed.append(event)

    ad._send_reply_request = fake_reply
    ad.handle_message = fake_handle
    return updates, routed


def _click(key, *, task_id="card-abc", req_id="evt-9", chattype="single",
           userid="ericyu", msgid="msg-1", chatid=None):
    detail = {"card_type": "button_interaction", "event_key": key, "task_id": task_id}
    body = {"msgid": msgid, "chattype": chattype, "from": {"userid": userid},
            "msgtype": "event",
            "event": {"eventtype": "template_card_event", "template_card_event": detail}}
    if chatid:
        body["chatid"] = chatid
    return {"cmd": m.APP_CMD_EVENT_CALLBACK, "headers": {"req_id": req_id}, "body": body}


def _notice(updates):
    """The template_card of the last update frame."""
    return updates[-1][2]["template_card"]


# ---------------------------------------------------------------------------
# 1. key parsing / redaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key, expected", [
    (KEY_APPROVE, ("ap-42", NONCE, "approve")),
    (KEY_REJECT, ("ap-42", NONCE, "reject")),
])
def test_valid_keys_parse(key, expected):
    assert iac.parse_iac_key(key) == expected


@pytest.mark.parametrize("key", [
    "iac:ap-42:approve",                    # three segments
    f"iac:ap-42:{NONCE}:approve:extra",     # five segments
    f"iac:ap-42:{NONCE}:cancel",            # unknown decision
    f"iac::{NONCE}:approve",                # empty approvalId
    "iac:ap-42::approve",                   # empty nonce
    f"iac:../../etc:{NONCE}:approve",       # path-shaped approvalId
    f"iac:ap 42:{NONCE}:approve",           # space in approvalId
    "opt0|批准",                             # not an iac key at all
])
def test_malformed_keys_are_rejected(key):
    assert iac.parse_iac_key(key) is None


def test_redaction_hides_the_nonce():
    assert iac.redact_iac_key(KEY_APPROVE) == "iac:ap-42:***:approve"
    assert NONCE not in iac.redact_iac_key(KEY_APPROVE)
    # anything malformed is redacted wholesale rather than echoed
    assert iac.redact_iac_key(f"iac:{NONCE}") == "iac:***"


def test_decide_url_needs_the_base_url(monkeypatch):
    assert iac.iac_decide_url("ap-42") == \
        "http://haro.internal:8000/api/assistant/runtime-api/iac/approvals/ap-42/decide"
    monkeypatch.delenv(iac.IAC_API_URL_ENV)
    assert iac.iac_decide_url("ap-42") is None


# ---------------------------------------------------------------------------
# 2. the decide request
# ---------------------------------------------------------------------------

def test_click_posts_the_decide_request():
    client = _FakeClient(_FakeResponse(200, {"cardText": "已批准", "status": "approved"}))
    ad = _adapter(client)
    updates, routed = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))

    call = client.calls[0]
    assert call["url"] == \
        "http://haro.internal:8000/api/assistant/runtime-api/iac/approvals/ap-42/decide"
    assert call["headers"]["Authorization"] == "Bearer rt-secret-token"
    assert call["json"] == {
        "decision": "approve", "wecomUserId": "ericyu", "nonce": NONCE,
        "key": KEY_APPROVE, "chatType": "single", "msgid": "msg-1", "taskId": "card-abc",
    }
    assert routed == []  # an iac click is never fed to the agent as a message


def test_reject_carries_its_own_decision():
    client = _FakeClient(_FakeResponse(200, {"cardText": "已拒绝"}))
    ad = _adapter(client)
    _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_REJECT)))
    assert client.calls[0]["json"]["decision"] == "reject"
    assert client.calls[0]["json"]["key"] == KEY_REJECT


# ---------------------------------------------------------------------------
# 3. the card rewrite
# ---------------------------------------------------------------------------

def test_200_card_text_is_written_back():
    ad = _adapter(_FakeClient(_FakeResponse(200, {"cardText": "已批准，等待第 2 位审批"})))
    updates, _ = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))

    req_id, cmd, body = updates[-1]
    assert (req_id, cmd) == ("evt-9", m.APP_CMD_RESPONSE_UPDATE)
    assert body["response_type"] == "update_template_card"
    card = body["template_card"]
    assert card["card_type"] == "text_notice"
    assert card["sub_title_text"] == "已批准，等待第 2 位审批"
    assert card["card_action"]["type"] == 1 and card["card_action"]["url"]


def test_4xx_card_text_is_written_back_too():
    ad = _adapter(_FakeClient(_FakeResponse(
        403, {"code": "iac_wecom_unbound", "message": "no", "cardText": "无权审批"})))
    updates, _ = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))
    assert _notice(updates)["sub_title_text"] == "无权审批"


@pytest.mark.parametrize("client", [
    _FakeClient(raises=RuntimeError("connection refused")),
    _FakeClient(_FakeResponse(502, {})),
])
def test_unreachable_or_5xx_writes_the_timeout_text(client):
    ad = _adapter(client)
    updates, _ = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))
    assert _notice(updates)["sub_title_text"] == iac.IAC_TEXT_UNAVAILABLE


def test_haro_timeout_writes_the_timeout_text(monkeypatch):
    monkeypatch.setattr(iac, "IAC_HTTP_TIMEOUT_SECONDS", 0.01)
    ad = _adapter(_FakeClient(_FakeResponse(200, {"cardText": "太晚了"}), delay=0.5))
    updates, _ = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))
    assert _notice(updates)["sub_title_text"] == iac.IAC_TEXT_UNAVAILABLE


def test_missing_haro_url_never_calls_out(monkeypatch):
    monkeypatch.delenv(iac.IAC_API_URL_ENV)
    client = _FakeClient(_FakeResponse(200, {"cardText": "已批准"}))
    ad = _adapter(client)
    updates, _ = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))
    assert client.calls == []
    assert _notice(updates)["sub_title_text"] == iac.IAC_TEXT_UNCONFIGURED


def test_card_url_from_the_pushed_card_is_reused_for_the_action():
    ad = _adapter(_FakeClient(_FakeResponse(200, {"cardText": "已批准"})))
    updates, _ = _wire(ad)
    ad._pending_button_cards["card-abc"] = {
        "chat_id": "ericyu", "title": "IaC 变更审批", "url": "https://haro.example/iac/ap-42",
        "options": [], "keys": [], "ts": 0.0, "consumed": False, "owner": "", "explicit": True}
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))
    card = _notice(updates)
    assert card["card_action"]["url"] == "https://haro.example/iac/ap-42"
    assert card["main_title"]["title"] == "IaC 变更审批"


# ---------------------------------------------------------------------------
# 4. group clicks and malformed keys
# ---------------------------------------------------------------------------

def test_group_click_is_refused_locally():
    client = _FakeClient(_FakeResponse(200, {"cardText": "已批准"}))
    ad = _adapter(client)
    updates, routed = _wire(ad)
    asyncio.run(ad._on_template_card_event(
        _click(KEY_APPROVE, chattype="group", chatid="group-7")))
    assert client.calls == []                              # Haro is never called
    assert _notice(updates)["sub_title_text"] == iac.IAC_TEXT_GROUP_FORBIDDEN
    assert routed == []


def test_group_click_detected_via_the_runtime_group_set():
    client = _FakeClient(_FakeResponse(200, {"cardText": "已批准"}))
    ad = _adapter(client)
    updates, _ = _wire(ad)
    ad._group_chat_ids.add("group-7")
    asyncio.run(ad._on_template_card_event(
        _click(KEY_APPROVE, chattype="single", chatid="group-7")))
    assert client.calls == []
    assert _notice(updates)["sub_title_text"] == iac.IAC_TEXT_GROUP_FORBIDDEN


def test_malformed_iac_key_is_ignored_with_a_warning(caplog):
    caplog.set_level(logging.WARNING, logger="plugins.platforms.wecom.adapter")
    client = _FakeClient(_FakeResponse(200, {"cardText": "已批准"}))
    ad = _adapter(client)
    updates, routed = _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(f"iac:ap-42:{NONCE}:cancel")))
    assert client.calls == [] and updates == [] and routed == []
    assert "malformed IaC approval key" in caplog.text
    assert NONCE not in caplog.text


# ---------------------------------------------------------------------------
# 5. non-iac keys keep the old behaviour
# ---------------------------------------------------------------------------

def test_non_iac_key_still_goes_through_the_button_registry():
    client = _FakeClient(_FakeResponse(200, {"cardText": "unused"}))
    ad = _adapter(client)
    updates, routed = _wire(ad)
    ad._pending_button_cards["btn-1"] = {
        "chat_id": "ericyu", "title": "请选择", "options": ["广州", "深圳"],
        "keys": ["opt0|广州", "opt1|深圳"], "ts": 0.0, "consumed": False, "owner": ""}
    asyncio.run(ad._on_template_card_event(_click("opt0|广州", task_id="btn-1")))
    assert client.calls == []                      # never touches the IaC path
    assert [event.text for event in routed] == ["广州"]


# ---------------------------------------------------------------------------
# 6. log redaction
# ---------------------------------------------------------------------------

def test_logs_redact_the_key_and_never_carry_the_nonce(caplog):
    caplog.set_level(logging.INFO, logger="plugins.platforms.wecom.adapter")
    ad = _adapter(_FakeClient(_FakeResponse(200, {"cardText": "已批准"})))
    _wire(ad)
    asyncio.run(ad._on_template_card_event(_click(KEY_APPROVE)))
    assert "iac:ap-42:***:approve" in caplog.text
    assert NONCE not in caplog.text
    assert "rt-secret-token" not in caplog.text


def test_group_refusal_log_is_redacted_too(caplog):
    caplog.set_level(logging.INFO, logger="plugins.platforms.wecom.adapter")
    ad = _adapter(_FakeClient(_FakeResponse(200, {"cardText": "x"})))
    _wire(ad)
    asyncio.run(ad._on_template_card_event(
        _click(KEY_REJECT, chattype="group", chatid="group-7")))
    assert "iac:ap-42:***:reject" in caplog.text
    assert NONCE not in caplog.text
