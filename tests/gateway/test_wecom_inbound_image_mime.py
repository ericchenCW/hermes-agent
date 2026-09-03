"""Inbound WeCom pictures served as application/octet-stream must still be cached as images."""
import asyncio
import os
import tempfile

os.environ.setdefault("HERMES_HOME", tempfile.mkdtemp())

from gateway.config import PlatformConfig
import plugins.platforms.wecom.adapter as m

WeComAdapter = [
    v for k, v in vars(m).items()
    if k.startswith("WeCom") and k.endswith("Adapter") and isinstance(v, type) and "Callback" not in k
][0]

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPG = b"\xff\xd8\xff\xe0" + b"\x00" * 64


def _adapter():
    return WeComAdapter(PlatformConfig(enabled=True, extra={"bot_id": "b", "secret": "s"}))


def _run(raw, content_type):
    ad = _adapter()

    async def fake_download(url, max_bytes=None):
        return raw, {"content-type": content_type}

    ad._download_remote_bytes = fake_download
    return asyncio.run(ad._cache_media("image", {"url": "https://wework.example/media?id=1"}))


def test_octet_stream_png_is_cached_as_png():
    path, mime = _run(PNG, "application/octet-stream")
    assert path.endswith(".png") and mime == "image/png"


def test_octet_stream_jpeg_is_cached_as_jpeg():
    path, mime = _run(JPG, "application/octet-stream")
    assert path.endswith((".jpg", ".jpeg")) and mime == "image/jpeg"


def test_explicit_image_header_still_honoured():
    path, mime = _run(PNG, "image/png")
    assert path.endswith(".png") and mime == "image/png"
