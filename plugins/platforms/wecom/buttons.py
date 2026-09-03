"""WeCom button cards (idcsre patch).

The model ends a reply with one line ``BUTTONS[标题]: 选项1 | 选项2 | ...``.  The adapter strips
that line from the visible text and sends a WeCom ``button_interaction`` template card after it
(passive reply on the turn's req_id, proactive send in DMs when no req_id is available).  A click
comes back as ``template_card_event``: within 5 s we answer with ``update_template_card`` (turning
the card into a text notice "已选择：X"), then feed the chosen label to the agent as a normal user
message."""

from __future__ import annotations

import asyncio
import logging
import os
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
BUTTON_LABEL_MAX = 10          # WeCom button text limit
BUTTON_TITLE_MAX = 26          # WeCom main_title.title limit
BUTTON_CARDS_MAX = 500         # registry hard cap
BUTTON_CARD_TTL_SECONDS = 24 * 3600
BUTTON_DEFAULT_TITLE = "请选择"
BUTTON_TRAILING_LINES = 4      # directive accepted within the last N lines
# text_notice cards must carry a card_action (errcode 42045 otherwise); the URL only matters if
# someone taps the "已选择" notice after a click.
BUTTON_CARD_ACTION_URL = os.environ.get("WECOM_CARD_ACTION_URL", "https://work.weixin.qq.com/")
BUTTON_PARTIAL_LINE_RE = re.compile(r"(?:^|\n)[ \t]*(?:\*\*)?BUTTONS(?![A-Za-z0-9])[^\n]*\Z", re.I)


class WeComButtonsMixin:
    """Button-card helpers mixed into WeComAdapter (uses its transport, policies and dedup)."""

    @staticmethod
    def _extract_button_directive(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Split a trailing ``BUTTONS[title]: a | b | c`` line off ``text``.

        Only the *last* non-blank line counts, and not when it sits inside an open triple-backtick
        fence (the skill docs quote the syntax verbatim).
        Returns (clean_text, spec) where spec = {"title", "options"} or None."""
        if not text or "BUTTONS" not in text.upper():
            return text, None
        # The directive is meant to be the last line, but the model often appends a "来源：..."
        # footer after it — accept it anywhere within the trailing few lines (outside code
        # fences / indented code).
        lines = text.rstrip().split("\n")
        m = found = None
        for idx in range(len(lines) - 1, max(-1, len(lines) - 1 - BUTTON_TRAILING_LINES), -1):
            line = lines[idx]
            if line.startswith("    ") or line.startswith("\t"):
                continue  # indented code block
            cand = BUTTON_DIRECTIVE_RE.fullmatch(line.strip())
            if not cand or "\n".join(lines[:idx]).count("```") % 2 == 1:
                continue  # no match, or inside an open code fence
            m, found = cand, idx
            break
        if m is None:
            return text, None
        del lines[found]
        head = "\n".join(lines)
        parts = [p.strip(" \t*`\"'“”") for p in re.split(r"\s*[|｜]\s*", m.group("opts"))]
        options: List[str] = []
        for part in parts:
            if part and part not in options:
                options.append(part)
        options = [p[:BUTTON_LABEL_MAX] for p in options[:BUTTON_MAX]]
        if not options:
            return text, None
        return head.rstrip(), {"title": (m.group("title") or BUTTON_DEFAULT_TITLE).strip()[:BUTTON_TITLE_MAX], "options": options}

    @staticmethod
    def _strip_partial_button_line(text: str) -> str:
        """Intermediate stream frames: hide a trailing (possibly half-written) ``BUTTONS...`` line
        so the directive never shows up in the bubble."""
        if not text or "BUTTONS" not in text.upper():
            return text
        stripped = text.rstrip()
        if m := BUTTON_PARTIAL_LINE_RE.search(stripped):
            return stripped[: m.start()].rstrip()
        clean, spec = WeComButtonsMixin._extract_button_directive(stripped)
        return clean if spec else text

    def _sweep_button_cards(self) -> None:
        cutoff = time.monotonic() - BUTTON_CARD_TTL_SECONDS
        for key, value in list(self._pending_button_cards.items()):
            if value["ts"] < cutoff:
                self._pending_button_cards.pop(key, None)
        overflow = len(self._pending_button_cards) - BUTTON_CARDS_MAX
        if overflow > 0:
            for key in sorted(self._pending_button_cards, key=lambda k: self._pending_button_cards[k]["ts"])[:overflow]:
                self._pending_button_cards.pop(key, None)

    def _build_button_card(self, chat_id: str, spec: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        task_id = f"btn-{uuid.uuid4().hex[:24]}"
        # the label rides inside the key so a click can be decoded even after a restart wiped the registry
        keys = [f"opt{i}|{label}" for i, label in enumerate(spec["options"])]
        card = {
            "card_type": "button_interaction",
            "main_title": {"title": spec["title"][:BUTTON_TITLE_MAX]},
            "task_id": task_id,
            "button_list": [
                {"text": label, "style": 1 if i == 0 else 4, "key": key}
                for i, (label, key) in enumerate(zip(spec["options"], keys))
            ],
        }
        self._sweep_button_cards()
        self._pending_button_cards[task_id] = {
            "chat_id": chat_id, "title": spec["title"], "options": list(spec["options"]),
            "keys": keys, "ts": time.monotonic(), "consumed": False,
        }
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

    async def _send_card_update(self, req_id: str, task_id: str, title: str, chosen_text: str) -> None:
        """Acknowledge a click by rewriting the card (must land within 5 s).

        First choice is WeCom's ``update_button`` (all buttons collapse into one disabled label);
        if the AI-bot channel rejects it, fall back to replacing the card with a ``text_notice`` —
        which WeCom requires to carry a ``card_action`` (errcode 42045 otherwise)."""
        attempts = [
            {"response_type": "update_button", "button": {"replace_name": chosen_text[:20]}},
            {
                "response_type": "update_template_card",
                "template_card": {
                    "card_type": "text_notice",
                    "main_title": {"title": title[:BUTTON_TITLE_MAX]},
                    "sub_title_text": chosen_text,
                    "card_action": {"type": 1, "url": BUTTON_CARD_ACTION_URL},
                    "task_id": task_id,
                },
            },
        ]
        for update in attempts:
            try:
                response = await self._send_reply_request(req_id, update, cmd=APP_CMD_RESPONSE_UPDATE, timeout=5.0)
                if not (errcode := int((response or {}).get("errcode", 0) or 0)):
                    logger.info("[%s] card update ok via %s task=%s", self.name, update["response_type"], task_id)
                    return
                logger.warning("[%s] card update via %s errcode=%s errmsg=%s", self.name, update["response_type"], errcode, str((response or {}).get("errmsg"))[:160])
            except Exception as exc:
                logger.warning("[%s] card update via %s failed: %s", self.name, update["response_type"], exc)

    async def _on_template_card_event_guarded(self, payload: Dict[str, Any]) -> None:
        try:
            await self._on_template_card_event(payload)
        except Exception as exc:
            logger.warning("[%s] template_card_event handling failed: %s", self.name, exc)

    async def _on_template_card_event(self, payload: Dict[str, Any]) -> None:
        """Button click: acknowledge by updating the card (<5 s), then hand the chosen label to
        the agent as if the user had typed it.

        Runs as its own task (never inline in the read loop — the update ack is delivered by it)."""
        body = payload.get("body") or {}
        req_id = self._payload_req_id(payload)
        event = body.get("event") if isinstance(body.get("event"), dict) else {}
        # WeCom may nest the detail under ``template_card_event``; the official example reads
        # ``event_key`` straight off ``event`` — accept both.
        detail = event.get("template_card_event") if isinstance(event.get("template_card_event"), dict) else event
        event_key = str(detail.get("event_key") or "").strip()
        task_id = str(detail.get("task_id") or "").strip()
        sender = body.get("from") if isinstance(body.get("from"), dict) else {}
        sender_id = str(sender.get("userid") or "").strip()
        self._sweep_button_cards()
        pending = self._pending_button_cards.get(task_id) or {}
        chat_id = str(pending.get("chat_id") or body.get("chatid") or sender_id).strip()
        is_group = chat_id in self._group_chat_ids or str(body.get("chattype") or "").lower() == "group"
        if pending and event_key in pending.get("keys", []):
            label = pending["options"][pending["keys"].index(event_key)]
        elif "|" in event_key:
            label = event_key.split("|", 1)[1].strip()
        else:
            label = event_key
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
        repeat = bool(pending.get("consumed"))
        logger.info("[%s] Button click: chat=%s sender=%s task=%s key=%r label=%r group=%s repeat=%s", self.name, chat_id, sender_id, task_id, event_key, label, is_group, repeat)
        # 0) policy first — a blocked user gets neither the update nor a routed message
        if is_group:
            self._group_chat_ids.add(chat_id)
            if not self._is_group_allowed(chat_id, sender_id):
                logger.info("[%s] Button click DROPPED by group policy: chat=%s", self.name, chat_id)
                return
        elif not self._is_dm_intake_allowed(sender_id):
            logger.info("[%s] Button click from %s blocked by DM policy", self.name, sender_id)
            return
        if not chat_id or not label:
            return  # malformed event: leave the card usable
        # Consume synchronously (no await yet): concurrent clicks on the same card are separate
        # tasks, and only the first one may route.
        if pending and not repeat:
            pending["consumed"] = True
            pending["chosen"] = label
        shown = (pending.get("chosen") if pending else None) or label or event_key
        # 1) acknowledge within 5 s — card becomes a text notice showing the (first) choice; sent as
        #    its own task so routing never waits on the ack
        if req_id and task_id:
            title = (pending.get("title") if pending else None) or BUTTON_DEFAULT_TITLE
            asyncio.ensure_future(self._send_card_update(req_id, task_id, title, f"已选择：{shown}"[:100]))
            await asyncio.sleep(0)  # let the update frame go out before routing
        if repeat:
            return
        msg_id = str(body.get("msgid") or f"btn-{task_id}-{event_key}")
        if self._dedup.is_duplicate(msg_id):
            return
        # 2) This turn has no usable inbound req_id: the event's req_id is spent on the update
        #    frame and the previous message's already served its reply.  Drop the stale one so the
        #    reply goes proactive (aibot_send_msg) instead of streaming on a dead req_id.
        self._last_chat_req_ids.pop(chat_id, None)
        self._stream_expired_chats.add(chat_id)
        self._button_click_chats.add(chat_id)
        source = self.build_source(chat_id=chat_id, chat_type="group" if is_group else "dm", user_id=sender_id or None, user_name=sender_id or None)
        event_obj = MessageEvent(
            text=label, message_type=MessageType.TEXT, source=source, raw_message=payload,
            message_id=msg_id, media_urls=[], media_types=[], timestamp=datetime.now(tz=timezone.utc),
        )
        try:
            await self.handle_message(event_obj)
        except Exception:
            # let the user click again: un-consume the card and forget the dedup id
            if pending:
                pending["consumed"] = False
                pending.pop("chosen", None)
            discard = getattr(self._dedup, "discard", None)
            if callable(discard):
                discard(msg_id)
            raise

    def _flow_log(self, turn, msg: str) -> None:  # [flow] instrumentation (hot patch)
        try:
            now = time.monotonic()
            if now - getattr(turn, "_flow_last_log", 0.0) >= 2.0:
                turn._flow_last_log = now
                logger.debug("[flow] adapter %s stream=%s", msg, getattr(turn, "stream_id", "?"))
        except Exception:
            pass
