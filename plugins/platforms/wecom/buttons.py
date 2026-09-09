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
from utils import env_bool
from plugins.platforms.wecom.media import APP_CMD_SEND

logger = logging.getLogger("plugins.platforms.wecom.adapter")

APP_CMD_RESPONSE_UPDATE = "aibot_respond_update_msg"   # update_template_card (event req_id, <5s)

# ``WECOM_BUTTONS`` gates the whole feature (default OFF: model instruction following is not
# reliable enough yet).  When off nothing is parsed into a card and none is sent; a directive line
# the model still emitted is stripped from the visible text so it never reaches the user.
BUTTONS_ENABLED = env_bool("WECOM_BUTTONS", False)
_BUTTONS_DISABLED_LOGGED = False
BUTTON_DIRECTIVE_RE = re.compile(
    r"^[ \t]*(?:\*\*)?BUTTONS(?:\[(?P<title>[^\]\n]{1,60})\])?(?:\*\*)?[：:][ \t]*(?P<opts>[^\n]+?)[ \t]*$",
    re.M | re.I,
)
BUTTON_MAX = 6
BUTTON_LABEL_MAX = 20          # option text cap (vote list shows it in full)
BUTTON_SHORT_WIDTH = 8         # display width (CJK=2) up to which a real button is used; longer → radio-list card
BUTTON_TITLE_MAX = 26          # WeCom main_title.title limit
BUTTON_CARDS_MAX = 500         # registry hard cap
BUTTON_CARD_TTL_SECONDS = 24 * 3600
BUTTON_DEFAULT_TITLE = "请选择"
BUTTON_TRAILING_LINES = 4      # directive accepted within the last N lines
BUTTON_STYLE = 4               # same style for every option (1 = primary blue reads as "selected")
# text_notice cards must carry a card_action (errcode 42045 otherwise); the URL only matters if
# someone taps the "已选择" notice after a click.
BUTTON_CARD_ACTION_URL = os.environ.get("WECOM_CARD_ACTION_URL", "https://work.weixin.qq.com/")
# Explicit (non-DSL) cards pushed through the ``platform_send`` verb: the caller
# hands over a structured spec, so nothing is parsed out of the reply text.
CARD_DESC_MAX = 76             # WeCom main_title.desc limit
CARD_TASK_ID_PREFIX = "card-"
BUTTON_PARTIAL_LINE_RE = re.compile(r"(?:^|\n)[ \t]*(?:\*\*)?BUTTONS(?![A-Za-z0-9])[^\n]*\Z", re.I)


class WeComButtonsMixin:
    """Button-card helpers mixed into WeComAdapter (uses its transport, policies and dedup)."""

    @staticmethod
    def _extract_button_directive(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Split a trailing ``BUTTONS[title]: a | b | c`` line off ``text``.

        Only the *last* non-blank line counts, and not when it sits inside an open triple-backtick
        fence (the skill docs quote the syntax verbatim).
        Returns (clean_text, spec) where spec = {"title", "options"} or None.

        With ``WECOM_BUTTONS`` off (the default) the directive is still parsed out of the text — so
        the raw line never shows up in the bubble — but no spec is returned, so no card is built or
        sent.  A directive-only reply degrades to its title as plain text."""
        clean, spec = WeComButtonsMixin._parse_button_directive(text)
        if spec and not BUTTONS_ENABLED:
            global _BUTTONS_DISABLED_LOGGED
            if not _BUTTONS_DISABLED_LOGGED:
                _BUTTONS_DISABLED_LOGGED = True
                logger.info("[wecom] buttons disabled, stripped BUTTONS directive from reply (WECOM_BUTTONS=0)")
            return (clean or spec["title"]), None
        return clean, spec

    @staticmethod
    def _parse_button_directive(text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Parsing half of :meth:`_extract_button_directive`, always active so intermediate stream
        frames hide the directive even when the feature is off."""
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
        clean, spec = WeComButtonsMixin._parse_button_directive(stripped)
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

    @staticmethod
    def _display_width(text: str) -> int:
        return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)

    def _build_button_card(self, chat_id: str, spec: Dict[str, Any], owner: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        task_id = f"btn-{uuid.uuid4().hex[:24]}"
        # the label rides inside the key so a click can be decoded even after a restart wiped the registry
        keys = [f"opt{i}|{label}" for i, label in enumerate(spec["options"])]
        title = spec["title"][:BUTTON_TITLE_MAX]
        main_title = {"title": title}
        if owner and chat_id in self._group_chat_ids:
            main_title["desc"] = "请提问者本人选择，其他人点击无效"
        if all(self._display_width(label) <= BUTTON_SHORT_WIDTH for label in spec["options"]):
            card = {
                "card_type": "button_interaction",
                "main_title": main_title,
                "task_id": task_id,
                "button_list": [
                    {"text": label, "style": BUTTON_STYLE, "key": key}
                    for label, key in zip(spec["options"], keys)
                ],
            }
        else:
            # WeCom ellipsizes button text after ~4 CJK chars ("开发…"); long options go into a
            # single-choice list that renders in full and comes back as
            # template_card_event.selected_items on submit.
            card = {
                "card_type": "vote_interaction",
                "main_title": main_title,
                "task_id": task_id,
                "checkbox": {
                    "question_key": "choice",
                    "mode": 0,
                    "option_list": [
                        {"id": key, "text": label, "is_checked": i == 0}
                        for i, (label, key) in enumerate(zip(spec["options"], keys))
                    ],
                },
                "submit_button": {"text": "确定", "key": "submit"},
            }
        self._sweep_button_cards()
        self._pending_button_cards[task_id] = {
            "chat_id": chat_id, "title": spec["title"], "options": list(spec["options"]),
            "keys": keys, "ts": time.monotonic(), "consumed": False, "owner": owner or "",
        }
        return task_id, card

    async def _send_button_card(self, chat_id: str, spec: Dict[str, Any], reply_req_id: Optional[str], owner: Optional[str] = None) -> bool:
        """Deliver a button card after the text: passive reply when a req_id is available
        (required in groups), proactive ``aibot_send_msg`` otherwise."""
        try:
            if not owner:
                owner = self._req_senders.get(reply_req_id or "") or (chat_id if chat_id not in self._group_chat_ids else None)
            task_id, card = self._build_button_card(chat_id, spec, owner)
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

    def _build_explicit_card(self, chat_id: str, card: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Structured spec → WeCom ``button_interaction`` template card.

        Unlike :meth:`_build_button_card` nothing is parsed out of reply text and
        the caller's keys ride through untouched (they are the caller's own
        capability tokens — the adapter neither decodes nor logs them). The card
        is registered so a later click can recover its title/url for the ack
        rewrite; the registry is the same bounded/TTL-swept dict.
        """
        task_id = f"{CARD_TASK_ID_PREFIX}{uuid.uuid4().hex[:24]}"
        title = str(card.get("title") or BUTTON_DEFAULT_TITLE)[:BUTTON_TITLE_MAX]
        main_title: Dict[str, Any] = {"title": title}
        if desc := str(card.get("desc") or "").strip():
            main_title["desc"] = desc[:CARD_DESC_MAX]
        buttons = []
        for entry in card.get("buttons") or []:
            style = entry.get("style")
            buttons.append({
                "text": str(entry.get("text") or "")[:BUTTON_LABEL_MAX],
                "style": style if isinstance(style, int) and not isinstance(style, bool) else BUTTON_STYLE,
                "key": str(entry.get("key") or ""),
            })
        built = {
            "card_type": "button_interaction",
            "main_title": main_title,
            "task_id": task_id,
            "button_list": buttons[:BUTTON_MAX],
        }
        self._sweep_button_cards()
        self._pending_button_cards[task_id] = {
            "chat_id": chat_id, "title": title, "options": [], "keys": [],
            "url": str(card.get("url") or ""), "ts": time.monotonic(),
            "consumed": False, "owner": "", "explicit": True,
        }
        return task_id, built

    async def send_card(self, chat_id: str, card: Dict[str, Any], fallback_text: str = ""):
        """Push one explicit interactive card to a DM (``platform_send`` card path).

        Proactive ``aibot_send_msg`` only: P1 serves single chats, where a
        proactive send is allowed (groups require a passive reply on a live
        req_id, which an external scheduler does not have). A card failure falls
        back to ``fallback_text`` as an ordinary message so the notification is
        never lost; the fallback never re-parses a BUTTONS directive.
        """
        from gateway.platforms.base import SendResult

        if not chat_id:
            return SendResult(success=False, error="chat_id is required")
        return await self._enqueue_chat_send(chat_id, lambda: self._send_card_inner(chat_id, card, fallback_text), is_control=True)

    async def _send_card_inner(self, chat_id: str, card: Dict[str, Any], fallback_text: str):
        from gateway.platforms.base import SendResult

        task_id = ""
        try:
            task_id, built = self._build_explicit_card(chat_id, card)
            response = await self._send_request(APP_CMD_SEND, {"chatid": chat_id, "msgtype": "template_card", "template_card": built})
            self._raise_for_wecom_error(response, "send card (proactive)")
            logger.info("[%s] Card sent task=%s buttons=%d", self.name, task_id, len(built["button_list"]))
            return SendResult(success=True, message_id=self._payload_req_id(response) or task_id, raw_response=response)
        except Exception as exc:
            logger.warning("[%s] Card delivery failed for chat %s: %s", self.name, chat_id, exc)
            self._pending_button_cards.pop(task_id, None)
            if not (fallback_text or "").strip():
                return SendResult(success=False, error=f"card delivery failed: {exc}")
            logger.info("[%s] Falling back to plain text for chat %s", self.name, chat_id)
            return await self._send_inner(chat_id, fallback_text, parse_directives=False)

    async def _send_card_update(self, req_id: str, task_id: str, title: str, chosen_text: str, user_id: str = "", url: str = "") -> None:
        """Acknowledge a click by rewriting the card (must land within 5 s).

        Field notes (2026-09-03): ``update_button`` is rejected by the AI-bot channel (40058);
        ``update_template_card`` with a ``text_notice`` needs a ``card_action`` (42045 otherwise).
        ``userids`` scopes the rewrite to the clicking user, as in the official SDK."""
        card = {
            "card_type": "text_notice",
            "main_title": {"title": title[:BUTTON_TITLE_MAX]},
            "sub_title_text": chosen_text,
            "card_action": {"type": 1, "url": url or BUTTON_CARD_ACTION_URL},
            "task_id": task_id,
        }
        attempts = [{"response_type": "update_template_card", "template_card": card}]
        if user_id:
            attempts.insert(0, {"response_type": "update_template_card", "template_card": card, "userids": [user_id]})
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
        # idcsre patch: IaC approval keys are decided by Haro, not by this
        # adapter's in-process registry — route them out before anything here
        # touches _pending_button_cards.
        from plugins.platforms.wecom.iac_approval import IAC_KEY_PREFIX
        if event_key.startswith(IAC_KEY_PREFIX):
            await self._on_iac_approval_click(
                payload, event_key=event_key, req_id=req_id, task_id=task_id,
                sender_id=sender_id, body=body,
            )
            return
        self._sweep_button_cards()
        pending = self._pending_button_cards.get(task_id) or {}
        chat_id = str(pending.get("chat_id") or body.get("chatid") or sender_id).strip()
        is_group = chat_id in self._group_chat_ids or str(body.get("chattype") or "").lower() == "group"
        def _decode(key: str) -> str:
            if pending and key in pending.get("keys", []):
                return pending["options"][pending["keys"].index(key)]
            return key.split("|", 1)[1].strip() if "|" in key else key

        chosen_ids: List[str] = []
        sel = detail.get("selected_items")
        if isinstance(sel, dict):
            for item in sel.get("selected_item") or []:
                if isinstance(item, dict):
                    oid = item.get("option_ids") or {}
                    chosen_ids += [str(x) for x in (oid.get("option_id") or [])] if isinstance(oid, dict) else []
        if chosen_ids:                      # vote/list card: submit button + selected option ids
            label = "、".join(_decode(k) for k in chosen_ids)
        else:                               # plain button card: the button key itself
            label = _decode(event_key)
        repeat = bool(pending.get("consumed"))
        owner = str(pending.get("owner") or "")
        logger.info("[%s] Button click: chat=%s sender=%s owner=%s task=%s key=%r label=%r group=%s repeat=%s", self.name, chat_id, sender_id, owner or "-", task_id, event_key, label, is_group, repeat)
        if owner and sender_id and sender_id != owner:
            # the card answers the asker's question: other members' clicks are ignored
            logger.info("[%s] Button click by %s ignored: card belongs to %s", self.name, sender_id, owner)
            return
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
            asyncio.ensure_future(self._send_card_update(req_id, task_id, title, f"已选择：{shown}"[:100], sender_id))
            await asyncio.sleep(0)  # let the update frame go out before routing
        if repeat:
            return
        msg_id = str(body.get("msgid") or f"btn-{task_id}-{event_key}")
        if self._dedup.is_duplicate(msg_id):
            return
        # 2) The event's req_id only accepts aibot_respond_update_msg — a stream/markdown reply on
        #    it comes back 846605 "invalid req_id" (verified 2026-09-03).  Drop the stale message
        #    req_id and mark the chat so this turn is buffered and delivered by proactive send
        #    (allowed in groups too via _button_click_chats).
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
