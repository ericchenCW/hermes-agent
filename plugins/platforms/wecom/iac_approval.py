"""IaC approval callback (idcsre patch).

Haro pushes an approval card through the ``platform_send`` verb with button keys
shaped ``iac:<approvalId>:<nonce>:approve|reject``. A click comes back as a
``template_card_event``; keys carrying the ``iac:`` prefix are handled here
instead of by the ordinary button-card registry, because the decision must be
made by Haro (which owns the nonce, the binding and the permission model) and
never by the adapter's in-process state — that state is lost on a restart, and
"lost" would mean "open to everyone".

The whole click has a 5 s budget: WeCom only accepts ``aibot_respond_update_msg``
on the event's own ``req_id`` within that window. Haro therefore gets 4 s and the
remaining second is reserved for the card rewrite.

Secrecy: the nonce is a one-time capability. It is never logged and never shown
to the user — every log line redacts the key to ``iac:<approvalId>:***:<decision>``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from plugins.platforms.wecom.buttons import BUTTON_DEFAULT_TITLE

try:  # pragma: no cover - exercised by the adapter's own dependency check
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger("plugins.platforms.wecom.adapter")

# --- key shape -------------------------------------------------------------
IAC_KEY_PREFIX = "iac:"
IAC_KEY_PARTS = 4
IAC_DECISION_APPROVE = "approve"
IAC_DECISION_REJECT = "reject"
IAC_DECISIONS = frozenset({IAC_DECISION_APPROVE, IAC_DECISION_REJECT})
# approvalId rides in the decide URL and the nonce in the body — both are kept to
# an opaque-token alphabet so neither can reshape the request path.
IAC_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
IAC_REDACTED = "***"

# --- Haro runtime API ------------------------------------------------------
IAC_API_URL_ENV = "HARO_API_URL"
IAC_TOKEN_ENV = "HARO_RUNTIME_TOKEN"
IAC_DECIDE_PATH = "/api/assistant/runtime-api/iac/approvals/{approval_id}/decide"
IAC_CHAT_TYPE_SINGLE = "single"

# --- budgets (the WeCom ack window is 5 s in total) ------------------------
IAC_CARD_UPDATE_BUDGET_SECONDS = 5.0
IAC_HTTP_TIMEOUT_SECONDS = 4.0

# --- user-facing card texts (fallbacks; Haro's own cardText wins) ----------
IAC_TEXT_GROUP_FORBIDDEN = "请在私聊处理"
IAC_TEXT_UNAVAILABLE = "处理超时，请稍后在 Haro 查看"
IAC_TEXT_UNCONFIGURED = "审批服务未配置，请在 Haro 处理"
IAC_TEXT_DONE = "已处理"
IAC_TEXT_FAILED = "处理失败，请在 Haro 查看"
IAC_CARD_TITLE = "IaC 审批"


def redact_iac_key(key: str) -> str:
    """``iac:<approvalId>:<nonce>:<decision>`` → ``iac:<approvalId>:***:<decision>``.

    Anything that is not a well-formed key is redacted wholesale rather than
    echoed: a malformed key may still carry a secret in the wrong position.
    """
    parts = str(key or "").split(":")
    if len(parts) != IAC_KEY_PARTS or parts[0] != IAC_KEY_PREFIX.rstrip(":"):
        return f"{IAC_KEY_PREFIX}{IAC_REDACTED}"
    return ":".join([parts[0], parts[1], IAC_REDACTED, parts[3]])


def parse_iac_key(key: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(approval_id, nonce, decision)``, or None when the shape is wrong."""
    parts = str(key or "").split(":")
    if len(parts) != IAC_KEY_PARTS or parts[0] != IAC_KEY_PREFIX.rstrip(":"):
        return None
    _, approval_id, nonce, decision = parts
    if not IAC_TOKEN_RE.match(approval_id) or not IAC_TOKEN_RE.match(nonce):
        return None
    if decision not in IAC_DECISIONS:
        return None
    return approval_id, nonce, decision


def iac_decide_url(approval_id: str) -> Optional[str]:
    """Decide endpoint for ``approval_id``, or None when ``$HARO_API_URL`` is unset."""
    base = (os.environ.get(IAC_API_URL_ENV) or "").strip().rstrip("/")
    if not base:
        return None
    return base + IAC_DECIDE_PATH.format(approval_id=approval_id)


class WeComIaCApprovalMixin:
    """IaC approval click handling, mixed into WeComAdapter."""

    def _iac_client(self):
        """Dedicated httpx client for the Haro runtime API.

        Deliberately NOT the adapter's SSRF-guarded media client: Haro is an
        operator-configured internal endpoint, which that guard exists to block.
        """
        client = getattr(self, "_iac_http_client", None)
        if client is None:
            if httpx is None:
                return None
            client = httpx.AsyncClient(timeout=IAC_HTTP_TIMEOUT_SECONDS)
            self._iac_http_client = client
        return client

    async def _close_iac_client(self) -> None:
        client = getattr(self, "_iac_http_client", None)
        self._iac_http_client = None
        if client is not None:
            try:
                await client.aclose()
            except Exception:  # pragma: no cover - teardown is best effort
                pass

    def _iac_card_context(self, task_id: str) -> Tuple[str, str]:
        """(title, card_action url) recovered from the pushed card, if still known."""
        pending = self._pending_button_cards.get(task_id) or {}
        return (str(pending.get("title") or IAC_CARD_TITLE), str(pending.get("url") or ""))

    async def _on_iac_approval_click(self, payload: Dict[str, Any], *, event_key: str,
                                     req_id: str, task_id: str, sender_id: str,
                                     body: Dict[str, Any]) -> None:
        """Route one ``iac:`` button click: Haro decides, we rewrite the card."""
        safe_key = redact_iac_key(event_key)
        parsed = parse_iac_key(event_key)
        if parsed is None:
            logger.warning("[%s] Ignoring malformed IaC approval key %s", self.name, safe_key)
            return
        approval_id, nonce, decision = parsed

        pending = self._pending_button_cards.get(task_id) or {}
        chat_id = str(pending.get("chat_id") or body.get("chatid") or sender_id).strip()
        is_group = chat_id in self._group_chat_ids or str(body.get("chattype") or "").lower() == "group"
        title, url = self._iac_card_context(task_id)
        ack_task_id = task_id or f"iac-{approval_id}"

        logger.info("[%s] IaC approval click: chat=%s sender=%s task=%s key=%s group=%s",
                    self.name, chat_id, sender_id, task_id or "-", safe_key, is_group)

        if is_group:
            # Refused locally: a group click cannot be attributed safely, and
            # bouncing off Haro would only cost a round trip out of the 5 s.
            self._group_chat_ids.add(chat_id)
            logger.info("[%s] IaC approval click in group refused key=%s", self.name, safe_key)
            await self._update_iac_card(req_id, ack_task_id, title, IAC_TEXT_GROUP_FORBIDDEN, sender_id, url)
            return

        card_text = await self._post_iac_decision(
            approval_id=approval_id, nonce=nonce, decision=decision, key=event_key,
            wecom_user_id=sender_id, msg_id=str(body.get("msgid") or ""), task_id=task_id,
            safe_key=safe_key,
        )
        await self._update_iac_card(req_id, ack_task_id, title, card_text, sender_id, url)

    async def _post_iac_decision(self, *, approval_id: str, nonce: str, decision: str,
                                 key: str, wecom_user_id: str, msg_id: str, task_id: str,
                                 safe_key: str) -> str:
        """Ask Haro to decide; return the card text to write back.

        Both 200 and 4xx carry Haro's own ``cardText`` — it is the authored,
        user-facing wording (已批准 / 链接已失效 / 无权审批 …) and is used as-is.
        Anything else (timeout, 5xx, unreachable, unconfigured) degrades to a
        local text and a WARN; the click is never left without an answer.
        """
        url = iac_decide_url(approval_id)
        if not url:
            logger.warning("[%s] IaC approval click but %s is unset key=%s",
                           self.name, IAC_API_URL_ENV, safe_key)
            return IAC_TEXT_UNCONFIGURED
        client = self._iac_client()
        if client is None:
            logger.warning("[%s] IaC approval click but no HTTP client is available key=%s",
                           self.name, safe_key)
            return IAC_TEXT_UNAVAILABLE

        headers = {"Content-Type": "application/json"}
        if token := (os.environ.get(IAC_TOKEN_ENV) or "").strip():
            headers["Authorization"] = f"Bearer {token}"
        else:
            logger.warning("[%s] %s is unset — calling the decide endpoint unauthenticated",
                           self.name, IAC_TOKEN_ENV)
        request_body: Dict[str, Any] = {
            "decision": decision,
            "wecomUserId": wecom_user_id,
            "nonce": nonce,
            "key": key,
            "chatType": IAC_CHAT_TYPE_SINGLE,
        }
        if msg_id:
            request_body["msgid"] = msg_id
        if task_id:
            request_body["taskId"] = task_id

        try:
            response = await asyncio.wait_for(
                client.post(url, json=request_body, headers=headers),
                timeout=IAC_HTTP_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("[%s] IaC decide timed out after %.1fs key=%s",
                           self.name, IAC_HTTP_TIMEOUT_SECONDS, safe_key)
            return IAC_TEXT_UNAVAILABLE
        except Exception as exc:
            logger.warning("[%s] IaC decide failed key=%s: %s: %s",
                           self.name, safe_key, type(exc).__name__, exc)
            return IAC_TEXT_UNAVAILABLE

        status = int(getattr(response, "status_code", 0) or 0)
        data: Dict[str, Any] = {}
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                data = parsed
        except Exception:
            logger.warning("[%s] IaC decide answered %s with a non-JSON body key=%s",
                           self.name, status, safe_key)
        card_text = str(data.get("cardText") or "").strip()
        if 200 <= status < 300:
            logger.info("[%s] IaC decide ok status=%s key=%s", self.name, status, safe_key)
            return card_text or IAC_TEXT_DONE
        if 400 <= status < 500:
            logger.info("[%s] IaC decide refused status=%s code=%s key=%s",
                        self.name, status, data.get("code") or "-", safe_key)
            return card_text or IAC_TEXT_FAILED
        logger.warning("[%s] IaC decide server error status=%s key=%s", self.name, status, safe_key)
        return IAC_TEXT_UNAVAILABLE

    async def _update_iac_card(self, req_id: str, task_id: str, title: str,
                               card_text: str, user_id: str, url: str = "") -> None:
        """Rewrite the card as a text notice — must land inside WeCom's 5 s window."""
        if not req_id or not task_id:
            logger.warning("[%s] Cannot acknowledge IaC click: req_id/task_id missing", self.name)
            return
        try:
            await asyncio.wait_for(
                self._send_card_update(req_id, task_id, title or BUTTON_DEFAULT_TITLE,
                                       card_text, user_id, url),
                timeout=IAC_CARD_UPDATE_BUDGET_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("[%s] IaC card update did not land within %.0fs task=%s",
                           self.name, IAC_CARD_UPDATE_BUDGET_SECONDS, task_id)
