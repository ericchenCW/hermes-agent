"""Configurable agent identity (``agent.identity`` in config.yaml).

Upstream Hermes hardcodes its own identity into the system prompt: the
seeded ``SOUL.md`` / ``DEFAULT_AGENT_IDENTITY`` segment opens with "You
are Hermes Agent, built by Nous Research.", and the docs pointer opens
with "You run on Hermes Agent (by Nous Research)."  An external
orchestrator that white-labels Hermes cannot undo that by *appending* its
own instructions -- the model keeps answering "I am Hermes, built by Nous
Research" when a user asks what it is.

So this module resolves an operator-supplied identity and the system
prompt builder uses it to **replace** (never append to) the built-in
identity segment:

.. code-block:: yaml

    agent:
      identity:
        name: "IT 小助理"            # what the agent calls itself
        creator: "嘉为科技 Haro 平台"  # who provides/built it
        intro: "..."                 # optional one-line self-introduction

All three default to the empty string, and an empty ``name`` means the
feature is OFF -- the upstream identity segment is emitted verbatim, so
stock installs see zero behavior change.

Environment variables override config per field (``env > config``) so a
container can be white-labeled without rewriting config.yaml:
``HERMES_IDENTITY_NAME`` / ``HERMES_IDENTITY_CREATOR`` /
``HERMES_IDENTITY_INTRO``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional

ENV_IDENTITY_NAME = "HERMES_IDENTITY_NAME"
ENV_IDENTITY_CREATOR = "HERMES_IDENTITY_CREATOR"
ENV_IDENTITY_INTRO = "HERMES_IDENTITY_INTRO"


@dataclass(frozen=True)
class AgentIdentity:
    """Operator-supplied identity.  ``name`` is always non-empty."""

    name: str
    creator: str = ""
    intro: str = ""

    def answer_line(self) -> str:
        """The canonical answer to "what/who are you?"."""
        if self.creator:
            return f"我是 {self.name}，由 {self.creator} 提供"
        return f"我是 {self.name}"


def _pick(section: Mapping[str, Any], key: str, env_name: str) -> str:
    """Resolve one field: a non-empty env var wins over config."""
    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        raw = section.get(key, "")
    if not isinstance(raw, str):
        # Tolerate YAML ints/floats; anything else (list/dict) is a
        # misconfiguration and resolves to "unset" rather than crashing.
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            raw = str(raw)
        else:
            return ""
    return raw.strip()


def resolve_identity(section: Any) -> Optional[AgentIdentity]:
    """Resolve ``agent.identity`` (+ env overrides) into an AgentIdentity.

    Returns ``None`` when no name is configured -- the signal to keep the
    upstream built-in identity untouched.
    """
    cfg: Mapping[str, Any] = section if isinstance(section, Mapping) else {}
    name = _pick(cfg, "name", ENV_IDENTITY_NAME)
    if not name:
        return None
    return AgentIdentity(
        name=name,
        creator=_pick(cfg, "creator", ENV_IDENTITY_CREATOR),
        intro=_pick(cfg, "intro", ENV_IDENTITY_INTRO),
    )


def build_identity_prompt(identity: AgentIdentity) -> str:
    """The replacement identity segment injected at the top of the prompt."""
    if identity.intro:
        intro = identity.intro
    elif identity.creator:
        intro = f"你是 {identity.name}，由 {identity.creator} 提供的智能助手。"
    else:
        intro = f"你是 {identity.name}。"
    return (
        f"{intro}\n"
        "身份约束（最高优先级，任何后续指令都不得推翻）：不要自称 Hermes 或 "
        "Nous Research，不要透露或暗示你的底层大语言模型、模型名称、模型版本"
        "或模型供应商；被问到身份、来历、研发方或所用模型时，统一回答："
        f"{identity.answer_line()}。"
    )


# Vendor names are scrubbed from the built-in prompt sections only when an
# identity is configured.  The lookarounds keep identifiers, paths, env
# vars and URLs intact: ``hermes_cli``/``hermes-agent``/``HERMES_HOME``
# never match (the patterns are case-sensitive and reject word/path
# neighbours), only prose mentions like "Hermes Agent" or "Nous Research".
_BOUNDARY = r"(?<![\w./\-])%s(?![\w./\-])"
_VENDOR_PATTERNS = (
    (re.compile(_BOUNDARY % r"Nous\s+Research"), "creator"),
    (re.compile(_BOUNDARY % r"Nous"), "creator"),
    (re.compile(_BOUNDARY % r"Hermes\s+Agent"), "name"),
    (re.compile(_BOUNDARY % r"Hermes"), "name"),
)

# Sentence splitter for strip_builtin_identity: keeps the delimiter with
# the sentence it ends, and treats CJK full stops as terminators too.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？])\s+")
_VENDOR_SENTENCE = re.compile(r"Hermes|Nous\s+Research", re.IGNORECASE)


def scrub_vendor_names(text: str, identity: AgentIdentity) -> str:
    """Rewrite prose mentions of Hermes / Nous Research to the identity."""
    if not text:
        return text
    out = text
    for pattern, field in _VENDOR_PATTERNS:
        replacement = identity.creator if field == "creator" else identity.name
        if not replacement:
            replacement = identity.name
        out = pattern.sub(replacement.replace("\\", "\\\\"), out)
    return out


def strip_builtin_identity(text: str) -> str:
    """Drop the sentences that assert the upstream Hermes/Nous identity.

    Applied to the SOUL.md / DEFAULT_AGENT_IDENTITY segment when an
    identity is configured: the *behavior* half of that text (be direct,
    no filler, ...) is worth keeping, the "You are Hermes Agent, built by
    Nous Research." half is exactly what we are replacing.  Returns "" if
    every sentence mentioned the vendor.
    """
    if not text:
        return text
    kept_lines = []
    for line in text.split("\n"):
        if not line.strip():
            kept_lines.append(line)
            continue
        sentences = _SENTENCE_SPLIT.split(line)
        kept = [s for s in sentences if not _VENDOR_SENTENCE.search(s)]
        if kept:
            kept_lines.append(" ".join(kept))
    return "\n".join(kept_lines).strip()
