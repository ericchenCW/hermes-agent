"""WeCom button cards (idcsre patch).

The model ends a reply with one line ``BUTTONS[标题]: 选项1 | 选项2 | ...``.  The adapter strips
that line from the visible text and sends a WeCom ``button_interaction`` template card after it
(passive reply on the turn's req_id, proactive send in DMs when no req_id is available).  A click
comes back as ``template_card_event``: within 5 s we answer with ``update_template_card`` (turning
the card into a text notice "已选择：X"), then feed the chosen label to the agent as a normal user
message."""

from __future__ import annotations

import logging
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.wecom.media import APP_CMD_SEND

logger = logging.getLogger("plugins.platforms.wecom.adapter")

APP_CMD_RESPONSE_UPDATE = "aibot_respond_update_msg"   # update_template_card (event req_id, <5s)

BUTTON_DIRECTIVE_RE = re.compile(
    r"^[ \t]*(?:\*\*)?BUTTONS(?:\[(?P<title>[^\]\n]{1,60})\])?(?:\*\*)?[：:][ \t]*(?P<opts>[^\n]+?)[ \t]*$",
    re.M | re.I,
)
BUTTON_MAX = 6
BUTTON_LABEL_MAX = 20
BUTTON_CARD_TTL_SECONDS = 24 * 3600
BUTTON_DEFAULT_TITLE = "请选择"


class WeComButtonsMixin:
    """Button-card helpers mixed into WeComAdapter (uses its transport, policies and dedup)."""

    @staticmethod
    def _extract_button_directive(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Split a trailing ``BUTTONS[title]: a | b | c`` line off ``text``.
        Returns (clean_text, spec) where spec = {"title", "options"} or None."""
        if not text or "BUTTONS" not in text.upper():
            return text, None
        matches = list(BUTTON_DIRECTIVE_RE.finditer(text))
        if not matches:
            return text, None
        m = matches[-1]
        parts = [p.strip(" \t*`\"'“”") for p in re.split(r"\s*[|｜]\s*", m.group("opts"))]
        options: List[str] = []
        for part in parts:
            part = part[:BUTTON_LABEL_MAX]
            if part and part not in options:
                options.append(part)
        options = options[:BUTTON_MAX]
        clean = (text[: m.start()] + text[m.end():]).rstrip()
        if not options:
            return clean, None
        return clean, {"title": (m.group("title") or BUTTON_DEFAULT_TITLE).strip()[:60], "options": options}

    def _build_button_card(self, chat_id: str, spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        task_id = f"btn-{uuid.uuid4().hex[:24]}"
        keys = [f"opt{i}" for i in range(len(spec["options"]))]
        card = {
            "card_type": "button_interaction",
            "main_title": {"title": spec["title"]},
            "task_id": task_id,
            "button_list": [
                {"text": label, "style": 1 if i == 0 else 4, "key": key}
                for i, (label, key) in enumerate(zip(spec["options"], keys))
            ],
        }
        self._pending_button_cards[task_id] = {
            "chat_id": chat_id, "title": spec["title"], "options": list(spec["options"]),
            "keys": keys, "ts": time.monotonic(),
        }
        cutoff = time.monotonic() - BUTTON_CARD_TTL_SECONDS  # keep the registry bounded
        for key, value in list(self._pending_button_cards.items()):
            if value["ts"] < cutoff:
                self._pending_button_cards.pop(key, None)
        return task_id, card

    async def _send_button_card(self, chat_id: str, spec: Dict[str, Any], reply_req_id: Optional[str]) -> bool:
        """Deliver a button card after the text: passive reply when a req_id is available
        (required in groups), proactive ``aibot_send_msg`` otherwise."""
        try:
            task_id, card = self._build_button_card(chat_id, spec)
            body = {"msgtype": "template_card", "template_card": card}
            if reply_req_id:
                try:
                    self._raise_for_wecom_error(await self._send_reply_request(reply_req_id, body), "send button card (reply)")
                    logger.info("[%s] Button card sent (reply) task=%s options=%s", self.name, task_id, spec["options"])
                    return True
                except Exception as exc:
                    logger.info("[%s] Button card reply failed (%s); trying proactive send", self.name, exc)
            self._raise_for_wecom_error(await self._send_request(APP_CMD_SEND, {"chatid": chat_id, **body}), "send button card (proactive)")
            logger.info("[%s] Button card sent (proactive) task=%s options=%s", self.name, task_id, spec["options"])
            return True
        except Exception as exc:
            logger.warning("[%s] Button card delivery failed for chat %s: %s", self.name, chat_id, exc)
            return False

    async def _on_template_card_event(self, payload: Dict[str, Any]) -> None:
        """Button click: acknowledge by updating the card (<5 s), then hand the chosen label to
        the agent as if the user had typed it."""
        body = payload.get("body") or {}
        req_id = self._payload_req_id(payload)
        event = body.get("event") if isinstance(body.get("event"), dict) else {}
        detail = event.get("template_card_event") if isinstance(event.get("template_card_event"), dict) else {}
        event_key = str(detail.get("event_key") or "").strip()
        task_id = str(detail.get("task_id") or "").strip()
        sender = body.get("from") if isinstance(body.get("from"), dict) else {}
        sender_id = str(sender.get("userid") or "").strip()
        chat_id = str(body.get("chatid") or sender_id).strip()
        is_group = str(body.get("chattype") or "").lower() == "group"
        pending = self._pending_button_cards.get(task_id) or {}
        label = event_key
        if pending and event_key in pending.get("keys", []):
            label = pending["options"][pending["keys"].index(event_key)]
        # selected_items (dropdown / multi-select) → append chosen option ids as text
        sel = detail.get("selected_items")
        if isinstance(sel, dict):
            ids: List[Any] = []
            for item in sel.get("selected_item") or []:
                if isinstance(item, dict):
                    oid = item.get("option_ids") or {}
                    ids += list(oid.get("option_id") or []) if isinstance(oid, dict) else []
            if ids:
                label = (label + " " if label else "") + ",".join(str(x) for x in ids)
        logger.info("[%s] Button click: chat=%s sender=%s task=%s key=%r label=%r group=%s", self.name, chat_id, sender_id, task_id, event_key, label, is_group)
        # 1) acknowledge within 5 s — card becomes a text notice with the choice
        if req_id and task_id:
            update = {
                "response_type": "update_template_card",
                "template_card": {
                    "card_type": "text_notice",
                    "main_title": {"title": ((pending.get("title") if pending else None) or BUTTON_DEFAULT_TITLE)[:60]},
                    "sub_title_text": f"已选择：{label or event_key}"[:160],
                    "task_id": task_id,
                },
            }
            try:
                response = await self._send_reply_request(req_id, update, cmd=APP_CMD_RESPONSE_UPDATE, timeout=5.0)
                if errcode := int((response or {}).get("errcode", 0) or 0):
                    logger.warning("[%s] update_template_card errcode=%s errmsg=%s", self.name, errcode, (response or {}).get("errmsg"))
            except Exception as exc:
                logger.warning("[%s] update_template_card failed: %s", self.name, exc)
        if not chat_id or not label:
            return
        # 2) policy + dedup, then route the label as a user message
        if is_group:
            self._group_chat_ids.add(chat_id)
            if not self._is_group_allowed(chat_id, sender_id):
                logger.info("[%s] Button click DROPPED by group policy: chat=%s", self.name, chat_id)
                return
        elif not self._is_dm_intake_allowed(sender_id):
            logger.info("[%s] Button click from %s blocked by DM policy", self.name, sender_id)
            return
        msg_id = str(body.get("msgid") or f"btn-{task_id}-{event_key}-{int(time.time())}")
        if self._dedup.is_duplicate(msg_id):
            return
        self._remember_reply_req_id(msg_id, req_id)
        if req_id:
            self._remember_chat_req_id(chat_id, req_id)
        self._pending_button_cards.pop(task_id, None)
        source = self.build_source(chat_id=chat_id, chat_type="group" if is_group else "dm", user_id=sender_id or None, user_name=sender_id or None)
        await self.handle_message(MessageEvent(
            text=label, message_type=MessageType.TEXT, source=source, raw_message=payload,
            message_id=msg_id, media_urls=[], media_types=[], timestamp=datetime.now(tz=timezone.utc),
        ))
