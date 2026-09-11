"""compose_sheet — build a numbered screenshot sheet from a knowledge-base document.

Wraps the vetted ``compose_from_doc.py`` script of the haro-kb-search skill so
read-only chat users can get step sheets without a shell.  The tool accepts
only a document path relative to the knowledge-base root plus a few
switches; it never takes a command line.  (The skill used to be named
canway-it-support-kb; that path is still probed for compatibility with
deployments that have not renamed it yet.)

Env (all optional):
  CANWAY_KB_ROOT          knowledge-base root   (default: auto-probed, falling back to /opt/data/kb/canway-it-support)
  COMPOSE_SHEET_SCRIPT    composing script      (default: auto-probed, falling back to the haro-kb-search skill path)
  COMPOSE_SHEET_PYTHON    interpreter           (default: the running interpreter)
  COMPOSE_SHEET_MAX_BYTES output image size ceiling (default: see WECOM_IMAGE_EFFECTIVE_MAX_BYTES below, minus 5% margin)

Post-processing: the script's own MEDIA:<path>.jpg output(s) are re-encoded with
Pillow when they exceed COMPOSE_SHEET_MAX_BYTES, so the gateway never has to
silently drop an oversized image (see _postprocess_media / _compress_jpeg).
"""
from __future__ import annotations

import glob
import io
import json
import os
import pathlib
import subprocess
import sys

from tools.registry import registry

DEFAULT_KB_ROOT = "/opt/data/kb/canway-it-support"
ALT_KB_ROOT = "/knowledge/canway-it-support"
ALT_KB_PARENT = "/knowledge"
HARO_SCRIPT = "/opt/data/skills/haro-kb-search/scripts/compose_from_doc.py"
DEFAULT_SCRIPT = "/opt/data/skills/canway-it-support-kb/scripts/compose_from_doc.py"
SKILLS_SCRIPT_GLOB = "/opt/data/skills/*/scripts/compose_from_doc.py"
MAX_STEPS = 40
TIMEOUT_SECONDS = 180

# 企业微信智能机器人图片大小上限。
# 官方文档「上传附件资源」(aibot_upload_media_init)写的技术上限是 10MB：
#   https://developer.work.weixin.qq.com/document/path/95098 （查证于 2026-09-11）
# 但现网 2026-09-11 生成的长图 2,054,583 B 通过该接口发送失败，历史发送成功的最大
# 长图仅 2,014,586 B —— 两者都逼近 2MB=2,097,152 B，与企业微信图片消息广泛引用的
# 2MB 上限吻合，说明「机器人回复图片」存在低于文档上传上限的实际生效限制。
# 保守按 2MB 取值，而非文档写的 10MB。
WECOM_IMAGE_DOC_MAX_BYTES = 10 * 1024 * 1024  # 文档「上传附件资源」技术上限，仅供参考
WECOM_IMAGE_EFFECTIVE_MAX_BYTES = 2 * 1024 * 1024  # 2,097,152 B，保守生效值（见上方说明）

# 压缩策略：JPEG 质量 85→60（步进 5）逐档尝试；仍超限则按最长边 0.85 倍逐步缩放，
# 缩放下限为最长边 1600px。
JPEG_QUALITY_STEPS = list(range(85, 55, -5))  # 85,80,75,70,65,60
MIN_LONG_EDGE = 1600
SCALE_STEP = 0.85


def _default_max_bytes() -> int:
    return int(WECOM_IMAGE_EFFECTIVE_MAX_BYTES * 0.95)


def _configured_max_bytes() -> int:
    env = os.environ.get("COMPOSE_SHEET_MAX_BYTES")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return _default_max_bytes()


COMPOSE_SHEET_MAX_BYTES = _configured_max_bytes()


def _kb_root() -> pathlib.Path:
    env = os.environ.get("CANWAY_KB_ROOT")
    if env:
        return pathlib.Path(env)
    for candidate in (DEFAULT_KB_ROOT, ALT_KB_ROOT):
        p = pathlib.Path(candidate)
        if p.is_dir():
            return p
    parent = pathlib.Path(ALT_KB_PARENT)
    if parent.is_dir():
        subdirs = [p for p in parent.iterdir() if p.is_dir()]
        if len(subdirs) == 1:
            return subdirs[0]
    return pathlib.Path(DEFAULT_KB_ROOT)


def _script() -> pathlib.Path:
    env = os.environ.get("COMPOSE_SHEET_SCRIPT")
    if env:
        return pathlib.Path(env)
    for candidate in (HARO_SCRIPT, DEFAULT_SCRIPT):
        p = pathlib.Path(candidate)
        if p.is_file():
            return p
    matches = sorted(glob.glob(SKILLS_SCRIPT_GLOB))
    if matches:
        return pathlib.Path(matches[0])
    return pathlib.Path(DEFAULT_SCRIPT)


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


def _compress_jpeg(src: pathlib.Path, max_bytes: int) -> tuple[pathlib.Path, bool]:
    """Re-encode ``src`` into a sibling ``<stem>-c.jpg`` file under ``max_bytes``.

    Tries JPEG_QUALITY_STEPS first, then progressively downscales (longest
    edge * SCALE_STEP, floor MIN_LONG_EDGE) at the lowest quality step.
    Never overwrites ``src``. Returns (dst_path, ok) where ok is False if the
    smallest result achieved is still over max_bytes (caller still gets that
    smallest file back rather than dropping the image).
    """
    from PIL import Image

    dst = src.with_name(src.stem + "-c" + src.suffix)
    best_bytes: bytes | None = None
    with Image.open(src) as im:
        im = im.convert("RGB")
        width, height = im.size

        def _try(candidate: "Image.Image", quality: int) -> bytes:
            buf = io.BytesIO()
            candidate.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue()

        for quality in JPEG_QUALITY_STEPS:
            data = _try(im, quality)
            if best_bytes is None or len(data) < len(best_bytes):
                best_bytes = data
            if len(data) <= max_bytes:
                dst.write_bytes(data)
                return dst, True

        scale = 1.0
        min_quality = JPEG_QUALITY_STEPS[-1]
        while int(width * scale) > MIN_LONG_EDGE or int(height * scale) > MIN_LONG_EDGE:
            scale *= SCALE_STEP
            new_w = max(1, int(width * scale))
            new_h = max(1, int(height * scale))
            resized = im.resize((new_w, new_h), Image.LANCZOS)
            data = _try(resized, min_quality)
            if best_bytes is None or len(data) < len(best_bytes):
                best_bytes = data
            if len(data) <= max_bytes:
                dst.write_bytes(data)
                return dst, True
            if new_w <= MIN_LONG_EDGE and new_h <= MIN_LONG_EDGE:
                break

    assert best_bytes is not None
    dst.write_bytes(best_bytes)
    return dst, len(best_bytes) <= max_bytes


def _postprocess_media(out: str, max_bytes: int = COMPOSE_SHEET_MAX_BYTES) -> str:
    """Compress any oversized MEDIA:<path>.jpg line(s) in the script output.

    Multiple MEDIA lines (paginated long images) are handled independently.
    Lines whose file is missing or already within the limit pass through
    unchanged. Never raises; a compression failure is reported as a WARNING
    line and the original MEDIA line is kept so no image is silently dropped.
    """
    lines = out.splitlines()
    result: list[str] = []
    for line in lines:
        if not line.startswith("MEDIA:"):
            result.append(line)
            continue
        path = pathlib.Path(line[len("MEDIA:") :].strip())
        try:
            size = path.stat().st_size
        except OSError:
            result.append(line)
            continue
        if size <= max_bytes:
            result.append(line)
            continue
        try:
            dst, ok = _compress_jpeg(path, max_bytes)
        except Exception as exc:  # pragma: no cover - defensive, Pillow failure is rare
            result.append(f"WARNING: 压缩失败（{exc}），已按原图 {size / 1024:.0f}KB 发送，" f"可能超过企业微信 {max_bytes / 1024:.0f}KB 限制")
            result.append(line)
            continue
        new_size = dst.stat().st_size
        result.append(f"已压缩：原 {size / 1024:.0f} KB → {new_size / 1024:.0f} KB")
        if not ok:
            result.append(f"WARNING: 压缩后仍为 {new_size / 1024:.0f} KB，超过 {max_bytes / 1024:.0f} KB 限制，已按可压缩到的最小结果发送")
        result.append(f"MEDIA:{dst}")
    return "\n".join(result)


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
    if not out:
        return "NO_IMAGES"
    return _postprocess_media(out, COMPOSE_SHEET_MAX_BYTES)


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
            "max": {"type": "integer", "minimum": 1, "maximum": MAX_STEPS, "description": f"Maximum images per sheet (default {MAX_STEPS}; a document's screenshots normally all fit on one sheet)"},
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
