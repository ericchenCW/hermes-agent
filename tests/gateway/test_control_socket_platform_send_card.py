"""``platform_send`` card path — the IaC approval push (P1).

The verb grew an optional ``card`` object ({title, desc?, buttons[], url?}) and
an optional ``chat_type`` (default "single"; only "single" is served in P1).
With a card the adapter's explicit ``send_card`` is used; without one the text
path stays exactly as it was, except that a ``BUTTONS[...]`` line in the body is
no longer allowed to become a card behind the caller's back.

Logging: exactly one INFO line per call, carrying the button COUNT — never a
button key (keys are one-time approval capabilities) and never the body.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from gateway.control_socket import (
    PLATFORM_SEND_CARD_BUTTONS_MAX,
    PLATFORM_SEND_CARD_DESC_MAX,
    PLATFORM_SEND_CARD_TITLE_MAX,
    PLATFORM_SEND_ERR_BAD_REQUEST,
    PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
    PLATFORM_SEND_NO_DIRECTIVES_FLAG,
    PLATFORM_SEND_VERB,
    VerbError,
    build_platform_send_handler,
)

SECRET = "机密正文：变更 3 台主机的安全组"
KEY_APPROVE = "iac:ap-42:9f3c7d1e:approve"
KEY_REJECT = "iac:ap-42:9f3c7d1e:reject"


def _card(**overrides):
    card = {
        "title": "IaC 变更审批",
        "desc": "变更 3 台主机的安全组",
        "buttons": [
            {"key": KEY_APPROVE, "text": "批准", "style": 1},
            {"key": KEY_REJECT, "text": "拒绝", "style": 2},
        ],
        "url": "https://haro.example/iac/ap-42",
    }
    card.update(overrides)
    return {k: v for k, v in card.items() if v is not None}


class _CardAdapter:
    """Adapter stand-in exposing both the text and the card entry point."""

    def __init__(self, *, message_id="card-1", card_raises=None):
        self.is_connected = True
        self._message_id = message_id
        self._card_raises = card_raises
        self.sent: list[tuple[str, str, dict | None]] = []
        self.cards: list[tuple[str, dict, str]] = []

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SimpleNamespace(success=True, message_id="text-1", error=None)

    async def send_card(self, chat_id, card, fallback_text=""):
        self.cards.append((chat_id, card, fallback_text))
        if self._card_raises is not None:
            raise self._card_raises
        return SimpleNamespace(success=True, message_id=self._message_id, error=None)


class _NoCardAdapter:
    """Text-only adapter: has no ``send_card`` at all."""

    is_connected = True

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SimpleNamespace(success=True, message_id="text-1", error=None)


def _run_handler(adapter_map, request, *, timeout=None):
    async def scenario():
        loop = asyncio.get_running_loop()
        handler = build_platform_send_handler(adapter_map.get, loop=loop, timeout=timeout)
        return await loop.run_in_executor(None, handler, request)

    return asyncio.run(scenario())


def _run_handler_error(adapter_map, request, *, timeout=None) -> VerbError:
    with pytest.raises(VerbError) as excinfo:
        _run_handler(adapter_map, request, timeout=timeout)
    return excinfo.value


def _request(**overrides):
    request = {"verb": PLATFORM_SEND_VERB, "platform": "wecom",
               "chat_id": "ericyu", "text": SECRET, "request_id": "iac-push-ap-42-1"}
    request.update(overrides)
    return request


# ---------------------------------------------------------------------------
# 1. Card happy path
# ---------------------------------------------------------------------------

def test_card_goes_to_send_card_and_returns_the_contract_result():
    adapter = _CardAdapter(message_id="msg-card-9")
    result = _run_handler({"wecom": adapter}, _request(card=_card()))

    assert result == {"message_id": "msg-card-9", "request_id": "iac-push-ap-42-1",
                      "platform": "wecom", "chat_id": "ericyu"}
    assert adapter.sent == []           # the text path was NOT used
    chat_id, card, fallback = adapter.cards[0]
    assert chat_id == "ericyu"
    assert fallback == SECRET           # text stays the fallback body
    assert [b["key"] for b in card["buttons"]] == [KEY_APPROVE, KEY_REJECT]
    assert card["title"] == "IaC 变更审批" and card["url"].endswith("/ap-42")


def test_chat_type_single_is_the_default_and_may_be_stated():
    adapter = _CardAdapter()
    _run_handler({"wecom": adapter}, _request(card=_card()))
    _run_handler({"wecom": adapter}, _request(card=_card(), chat_type="SINGLE"))
    assert len(adapter.cards) == 2


def test_optional_card_fields_may_be_omitted():
    adapter = _CardAdapter()
    _run_handler({"wecom": adapter},
                 _request(card={"title": "T", "buttons": [{"key": KEY_APPROVE, "text": "批准"}]}))
    _, card, _ = adapter.cards[0]
    assert card == {"title": "T", "buttons": [{"key": KEY_APPROVE, "text": "批准"}]}


# ---------------------------------------------------------------------------
# 2. bad_request — card schema and chat_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("card, needle", [
    ("not-an-object", "card must be an object"),
    ({"buttons": [{"key": "k", "text": "t"}]}, "card.title"),
    ({"title": "", "buttons": [{"key": "k", "text": "t"}]}, "card.title"),
    ({"title": "x" * (PLATFORM_SEND_CARD_TITLE_MAX + 1),
      "buttons": [{"key": "k", "text": "t"}]}, "card.title exceeds"),
    ({"title": "T", "desc": "d" * (PLATFORM_SEND_CARD_DESC_MAX + 1),
      "buttons": [{"key": "k", "text": "t"}]}, "card.desc exceeds"),
    ({"title": "T"}, "card.buttons is required"),
    ({"title": "T", "buttons": []}, "card.buttons must hold"),
    ({"title": "T", "buttons": [{"key": f"k{i}", "text": "t"}
                                for i in range(PLATFORM_SEND_CARD_BUTTONS_MAX + 1)]},
     "card.buttons must hold"),
    ({"title": "T", "buttons": [{"text": "t"}]}, "buttons[0].key"),
    ({"title": "T", "buttons": [{"key": "k"}]}, "buttons[0].text"),
    ({"title": "T", "buttons": [{"key": "k", "text": "t"}, {"key": "k", "text": "u"}]},
     "duplicated"),
    ({"title": "T", "buttons": [{"key": "k", "text": "t", "style": {"a": 1}}]},
     "buttons[0].style"),
    ({"title": "T", "buttons": [{"key": "k", "text": "t", "colour": "red"}]},
     "unknown field"),
    ({"title": "T", "buttons": [{"key": "k", "text": "t"}], "owners": ["a"]},
     "unknown field"),
])
def test_malformed_card_is_bad_request(card, needle):
    error = _run_handler_error({"wecom": _CardAdapter()}, _request(card=card))
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
    assert needle in error.detail


def test_zero_buttons_and_seven_buttons_are_both_rejected():
    for count in (0, PLATFORM_SEND_CARD_BUTTONS_MAX + 1):
        card = {"title": "T", "buttons": [{"key": f"k{i}", "text": "t"} for i in range(count)]}
        error = _run_handler_error({"wecom": _CardAdapter()}, _request(card=card))
        assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
        assert f"(got {count})" in error.detail


def test_group_chat_type_is_refused_in_p1():
    adapter = _CardAdapter()
    error = _run_handler_error({"wecom": adapter}, _request(card=_card(), chat_type="group"))
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
    assert "not supported" in error.detail and "group" in error.detail
    assert adapter.cards == [] and adapter.sent == []


def test_group_chat_type_is_refused_for_plain_text_too():
    error = _run_handler_error({"wecom": _CardAdapter()}, _request(chat_type="group"))
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST


@pytest.mark.parametrize("chat_type", ["", "  ", "channel", 7])
def test_unusable_chat_type_is_bad_request(chat_type):
    error = _run_handler_error({"wecom": _CardAdapter()}, _request(chat_type=chat_type))
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST


def test_card_on_an_adapter_without_send_card_is_platform_unavailable():
    error = _run_handler_error({"wecom": _NoCardAdapter()}, _request(card=_card()))
    assert error.code == PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE
    assert "cannot send cards" in error.detail


def test_text_is_still_required_alongside_a_card():
    error = _run_handler_error({"wecom": _CardAdapter()}, _request(text="", card=_card()))
    assert error.code == PLATFORM_SEND_ERR_BAD_REQUEST
    assert "text is required" in error.detail


# ---------------------------------------------------------------------------
# 3. Text path unchanged, and the BUTTONS DSL no longer fires on it
# ---------------------------------------------------------------------------

def test_without_a_card_the_text_path_is_used():
    adapter = _CardAdapter()
    result = _run_handler({"wecom": adapter}, _request())
    assert result["message_id"] == "text-1"
    assert adapter.cards == []
    chat_id, content, metadata = adapter.sent[0]
    assert (chat_id, content) == ("ericyu", SECRET)
    assert metadata == {PLATFORM_SEND_NO_DIRECTIVES_FLAG: True}


def test_buttons_directive_in_text_does_not_trigger_a_card():
    adapter = _CardAdapter()
    body = "请选择：\nBUTTONS[请选择]: 批准 | 拒绝"
    _run_handler({"wecom": adapter}, _request(text=body))
    assert adapter.cards == []                       # no card path
    chat_id, content, metadata = adapter.sent[0]
    assert content == body                           # delivered verbatim
    assert metadata[PLATFORM_SEND_NO_DIRECTIVES_FLAG] is True


def test_adapters_that_reject_metadata_still_get_the_plain_call():
    class _LegacyAdapter:
        is_connected = True

        def __init__(self):
            self.sent = []

        async def send(self, chat_id, content):
            self.sent.append((chat_id, content))
            return SimpleNamespace(success=True, message_id="legacy-1", error=None)

    adapter = _LegacyAdapter()
    result = _run_handler({"wecom": adapter}, _request())
    assert result["message_id"] == "legacy-1"
    assert adapter.sent == [("ericyu", SECRET)]


# ---------------------------------------------------------------------------
# 4. Logging: counts, never keys, never the body
# ---------------------------------------------------------------------------

def test_log_line_carries_the_button_count_but_no_key_and_no_body(caplog):
    caplog.set_level(logging.INFO, logger="gateway.control_socket")
    _run_handler({"wecom": _CardAdapter()}, _request(card=_card()))
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "card_buttons=2" in logged and "chat_type=single" in logged
    assert "result=ok" in logged and "request_id=iac-push-ap-42-1" in logged
    for forbidden in (KEY_APPROVE, KEY_REJECT, "9f3c7d1e", SECRET, "IaC 变更审批",
                      "变更 3 台主机的安全组", "haro.example"):
        assert forbidden not in logged


def test_bad_card_log_line_does_not_leak_the_keys(caplog):
    caplog.set_level(logging.INFO, logger="gateway.control_socket")
    _run_handler_error(
        {"wecom": _CardAdapter()},
        _request(card=_card(buttons=[{"key": KEY_APPROVE, "text": "批准"},
                                     {"key": KEY_APPROVE, "text": "拒绝"}])),
    )
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert PLATFORM_SEND_ERR_BAD_REQUEST in logged
    assert KEY_APPROVE not in logged and SECRET not in logged
