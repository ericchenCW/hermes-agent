"""compose_sheet — build a numbered screenshot sheet from a knowledge-base document.

Wraps the vetted ``compose_from_doc.py`` script of the canway-it-support-kb
skill so read-only chat users can get step sheets without a shell.  The tool
accepts only a document path relative to the knowledge-base root plus a few
switches; it never takes a command line.

Env (all optional):
  CANWAY_KB_ROOT          knowledge-base root   (default /opt/data/kb/canway-it-support)
  COMPOSE_SHEET_SCRIPT    composing script      (default /opt/data/skills/canway-it-support-kb/scripts/compose_from_doc.py)
  COMPOSE_SHEET_PYTHON    interpreter           (default: the running interpreter)
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from tools.registry import registry

DEFAULT_KB_ROOT = "/opt/data/kb/canway-it-support"
DEFAULT_SCRIPT = "/opt/data/skills/canway-it-support-kb/scripts/compose_from_doc.py"
MAX_STEPS = 8
TIMEOUT_SECONDS = 180


def _kb_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CANWAY_KB_ROOT") or DEFAULT_KB_ROOT)


def _script() -> pathlib.Path:
    return pathlib.Path(os.environ.get("COMPOSE_SHEET_SCRIPT") or DEFAULT_SCRIPT)


def resolve_doc(doc: str) -> pathlib.Path:
    """Map a KB-relative document path to a real file, refusing escapes."""
    rel = str(doc or "").strip().lstrip("/")
    if not rel or rel.endswith("/") or "\\" in rel or "\x00" in rel:
        raise ValueError("doc must be a markdown file path relative to the knowledge base, e.g. guides/access/vpn-user-guide.md")
    root = _kb_root().resolve()
    target = (root / rel).resolve()
    if root not in target.parents:
        raise ValueError("doc must stay inside the knowledge base")
    if target.suffix.lower() != ".md" or not target.is_file():
        raise ValueError(f"document not found: {rel}")
    return target


def compose_sheet(doc: str, os_name: str | None = None, section: str | None = None, max_steps: int | None = None) -> str:
    try:
        target = resolve_doc(doc)
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    script = _script()
    if not script.is_file():
        return json.dumps({"error": f"composing script missing: {script}"}, ensure_ascii=False)
    n = MAX_STEPS if max_steps is None else max(1, min(MAX_STEPS, int(max_steps)))
    cmd = [os.environ.get("COMPOSE_SHEET_PYTHON") or sys.executable, str(script), str(target.relative_to(_kb_root().resolve())), "--max", str(n)]
    if os_name in ("windows", "mac"):
        cmd += ["--os", os_name]
    if section:
        cmd += ["--section", str(section)[:60]]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS, cwd=str(_kb_root()))
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"compose timed out after {TIMEOUT_SECONDS}s"}, ensure_ascii=False)
    out = (proc.stdout or "").strip()
    if proc.returncode != 0:
        return json.dumps({"error": "compose failed", "detail": ((proc.stderr or "") + out)[-800:]}, ensure_ascii=False)
    return out or "NO_IMAGES"


COMPOSE_SHEET_SCHEMA = {
    "name": "compose_sheet",
    "description": (
        "Compose one numbered step sheet (JPG) from the screenshots of a knowledge-base document. "
        "Pass the document path relative to the knowledge base (as listed in the index files, e.g. "
        "guides/access/vpn-user-guide.md). Output: a '图 N: caption' mapping followed by a final "
        "'MEDIA:/opt/data/cache/sheets/....jpg' line to paste at the end of the reply, or NO_IMAGES. "
        "Use os='windows'|'mac' when the document has per-OS sections; section=keyword to keep one chapter."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "doc": {"type": "string", "description": "Document path relative to the knowledge base root (must end with .md)"},
            "os": {"type": "string", "enum": ["windows", "mac"], "description": "Keep only this OS's chapter plus common sections"},
            "section": {"type": "string", "description": "Keep only chapters whose heading contains this keyword"},
            "max": {"type": "integer", "minimum": 1, "maximum": MAX_STEPS, "description": f"Maximum steps on the sheet (default {MAX_STEPS})"},
        },
        "required": ["doc"],
    },
}


def check_compose_sheet_requirements() -> bool:
    return _script().is_file() and _kb_root().is_dir()


registry.register(
    name="compose_sheet",
    toolset="compose_sheet",
    schema=COMPOSE_SHEET_SCHEMA,
    handler=lambda args, **kw: compose_sheet(
        args.get("doc", ""), args.get("os"), args.get("section"), args.get("max"),
    ),
    check_fn=check_compose_sheet_requirements,
    description="Compose a numbered screenshot sheet from a knowledge-base document",
    emoji="🖼️",
)
