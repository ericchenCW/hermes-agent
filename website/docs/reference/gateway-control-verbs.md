---
title: "Gateway Control Verbs"
description: "The gateway control socket's action verbs — platform_send for outbound text and interactive cards."
---

# Gateway Control Verbs

The gateway owns a local-only control socket (`$HERMES_HOME/gateway.sock`, or a
named pipe on Windows) that answers one JSON request per connection. Beside the
liveness verbs (`identify`, `status`) it serves *action* verbs, which perform
something that can fail on its own and therefore answer with an error code.

Filesystem/pipe ACLs are the auth boundary — the socket is never a TCP port.

## `platform_send`

An external scheduler (Haro's cron jobs, the IaC approval pipeline) runs outside
the gateway process but must deliver into a live IM session. Only the process
holding the platform connection may send — a second WeCom websocket kicks the
gateway offline — so the gateway performs the send on the caller's behalf and
answers synchronously.

### Request

```json
{"verb": "platform_send", "id": 1, "protocol": 1,
 "platform": "wecom",
 "chat_id": "ericyu",
 "text": "IaC 变更待审批：变更 3 台主机的安全组",
 "chat_type": "single",
 "card": {
   "title": "IaC 变更审批",
   "desc": "变更 3 台主机的安全组",
   "buttons": [{"key": "iac:ap-42:<nonce>:approve", "text": "批准", "style": 1},
               {"key": "iac:ap-42:<nonce>:reject",  "text": "拒绝", "style": 2}],
   "url": "https://haro.example/iac/ap-42"
 },
 "request_id": "iac-push-ap-42-1"}
```

| Field | Required | Notes |
| --- | --- | --- |
| `platform` | yes | Only `wecom` is served. |
| `chat_id` | yes | WeCom `chatid` (a userid in a single chat). |
| `text` | yes | ≤ 4000 chars. Delivered **verbatim** — no templating, no command interpretation, and a trailing `BUTTONS[...]` line is *not* turned into a card. With a `card` it is the fallback body. |
| `chat_type` | no | Defaults to `single`. `group` is refused (`bad_request`) — a card cannot be pushed proactively into a group, and a group click cannot be attributed safely. |
| `card` | no | Interactive card; see below. Requires an adapter exposing `send_card`. |
| `request_id` | no | Opaque correlation token, echoed back and logged. Never validated. |

### The `card` object

| Field | Required | Limit |
| --- | --- | --- |
| `title` | yes | ≤ 128 chars (WeCom truncates to 26). |
| `desc` | no | ≤ 512 chars (WeCom truncates to 76). |
| `buttons` | yes | 1–6 entries, each `{key, text, style?}`; keys must be unique. |
| `buttons[].key` | yes | ≤ 256 chars. Passed through **untouched** — it is the caller's own capability token. |
| `buttons[].text` | yes | ≤ 64 chars (WeCom truncates to 20). |
| `buttons[].style` | no | Integer or string; defaults to the adapter's neutral style. |
| `url` | no | ≤ 1024 chars. Used as the `card_action` target of the acknowledgement card after a click. |

Unknown fields are rejected rather than ignored, so a caller never believes it
sent something the gateway dropped.

A card is delivered as a WeCom `button_interaction` template card over
`aibot_send_msg`. If the card cannot be delivered, the adapter falls back to
sending `text` as an ordinary message.

### Answers

```json
{"ok": true,  "protocol": 1, "id": 1,
 "result": {"message_id": "…", "request_id": "…", "platform": "wecom", "chat_id": "…"}}
{"ok": false, "protocol": 1, "id": 1, "error": "<code>: <detail>"}
```

| Code | Meaning |
| --- | --- |
| `bad_request` | Schema violation: unknown platform, missing `text`, malformed `card`, unsupported `chat_type`. |
| `platform_unavailable` | No live adapter, adapter disconnected, no gateway loop, or a `card` for an adapter with no `send_card`. |
| `rate_limited` | WeCom errcode `846607` (30 msgs/min/chat). |
| `send_failed` | The adapter reported a failure. |
| `timeout` | The send did not finish within the budget (`$HERMES_PLATFORM_SEND_TIMEOUT`, default 15 s). |

### Logging

Exactly one INFO line per call, carrying platform, chat id, text **length**,
chat type, button **count**, request id and outcome. Never the message body and
never a button key — keys are one-time approval capabilities.
