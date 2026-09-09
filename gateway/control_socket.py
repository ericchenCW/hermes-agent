"""
Gateway control socket — the gateway-owned local coordination surface.

Migration step 1 of the #92091 design: every other process on the machine
(the updater, `hermes serve`/dashboard, the Desktop app) currently discovers
gateway identity/state by scanning the process table and string-matching argv
or by reading ``gateway_state.json`` (which can outlive its writer). This
module gives the gateway an OWNED contract instead: a local-only socket the
gateway process creates at startup and removes on clean shutdown, answering
versioned JSON verbs. A connectable socket with a well-formed ``identify``
answer IS liveness — no PID-reuse heuristics.

v1 verbs (observation only — no behavior change for the gateway):

- ``identify`` → pid, profile label, hermes_home, code_sha/code_version
  (the #91283 stamps, now queryable live), supervisor kind, served profiles,
  start_time, protocol version.
- ``status``   → the live runtime-status payload (what ``gateway_state.json``
  holds today, but answered by the process itself, race-free).

Action verbs wired by the gateway runner:

- ``pause-for-update`` → drain and exit cleanly for an update.
- ``platform_send``    → send one text message to a chat through a live
  platform adapter. External schedulers (Haro's cron jobs) cannot open their
  own platform connection — a second WeCom WS kicks the gateway offline — so
  the gateway performs the send on their behalf and answers synchronously.

Transport:

- POSIX: Unix domain socket at ``$HERMES_HOME/gateway.sock``. When the home
  path is too long for ``sun_path`` (~104 bytes on macOS/BSD), the socket is
  bound in the system temp dir and a pointer file
  ``$HERMES_HOME/gateway.sock.path`` records the real location; clients
  follow the pointer transparently.
- Windows: named pipe ``\\\\.\\pipe\\hermes-gateway-<home-hash>`` served via
  the proactor event loop. Same trust model (per-user namespace).

Never a TCP port. Filesystem/pipe ACLs are the auth boundary — the same
trust model as ``gateway_state.json`` today.

v1 wire contract: ONE request per connection — a single JSON line in, a
single JSON line out, then the server closes. Clients must not rely on
keep-alive or pipelining. Verb handlers may touch disk (they run in an
executor server-side) but must stay fast; the client budget is small.

Consumers (``hermes update --plan`` inventory, the post-update fleet version
matrix) PREFER the socket when it answers and fall back to the existing
state-file/scan layer when it doesn't — old gateways mid-upgrade and crashed
processes keep working exactly as before. The scan layer is demoted, not
deleted.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import hashlib
import inspect
import json
import logging
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

CONTROL_PROTOCOL_VERSION = 1

_SOCKET_FILENAME = "gateway.sock"
_POINTER_FILENAME = "gateway.sock.path"
_IS_WINDOWS = sys.platform == "win32"

# Practical sun_path limit: 104 on macOS/BSD, 108 on Linux. Stay under the
# smaller bound with margin for the NUL terminator.
_MAX_UNIX_PATH = 100

# Requests and responses are single JSON lines. Bound them so a misbehaving
# peer can't balloon gateway memory.
_MAX_REQUEST_BYTES = 64 * 1024
_MAX_RESPONSE_BYTES = 512 * 1024

_DEFAULT_CLIENT_TIMEOUT = 2.0

# ---------------------------------------------------------------------------
# platform_send verb (outbound push for external schedulers)
# ---------------------------------------------------------------------------
#
# Haro's scheduled jobs run OUTSIDE the gateway process but must deliver their
# result into a live IM session. Only the process holding the platform
# connection may send (a second process opening its own WeCom WS kicks the
# gateway offline), so the gateway exposes the send as a control verb: the
# caller hands over {platform, chat_id, text} and blocks for the outcome.
#
# The wire contract (agreed with the Haro side, 2026-09-10):
#
#   request  {"verb":"platform_send","id":<any>,"protocol":1,
#             "platform":"wecom","chat_id":"…","text":"…",
#             "request_id":"<uuid, echoed only>"}
#   success  {"ok":true,"protocol":1,"id":…,
#             "result":{"message_id":"…","request_id":"…",
#                       "platform":"wecom","chat_id":"…"}}
#   failure  {"ok":false,"protocol":1,"id":…,"error":"<code>: <detail>"}
#
# Failures use the framework's own failure envelope (no ``result`` key); the
# machine-readable part is the code prefix of ``error``.
#
# Error codes are constants so the contract can be tuned in one place.
PLATFORM_SEND_VERB = "platform_send"
PLATFORM_SEND_MAX_TEXT_CHARS = 4000
PLATFORM_SEND_DEFAULT_TIMEOUT = 15.0
PLATFORM_SEND_TIMEOUT_ENV = "HERMES_PLATFORM_SEND_TIMEOUT"

# Platforms this verb will attempt at all. Anything else is a caller mistake
# (bad_request), not a transient gateway condition (platform_unavailable).
PLATFORM_SEND_SUPPORTED_PLATFORMS = frozenset({"wecom"})

PLATFORM_SEND_ERR_BAD_REQUEST = "bad_request"
PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE = "platform_unavailable"
PLATFORM_SEND_ERR_RATE_LIMITED = "rate_limited"
PLATFORM_SEND_ERR_SEND_FAILED = "send_failed"
PLATFORM_SEND_ERR_TIMEOUT = "timeout"

PLATFORM_SEND_ERR_CODES = frozenset({
    PLATFORM_SEND_ERR_BAD_REQUEST,
    PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
    PLATFORM_SEND_ERR_RATE_LIMITED,
    PLATFORM_SEND_ERR_SEND_FAILED,
    PLATFORM_SEND_ERR_TIMEOUT,
})

# WeCom's "too many messages" errcode. Recognised in the adapter's error text
# (SendResult.error) or in an adapter exception, and reported as its own code
# so the caller can back off instead of treating it as a hard failure.
PLATFORM_SEND_RATE_LIMIT_ERRCODE = "846607"


# ---------------------------------------------------------------------------
# Path resolution (shared by server and client)
# ---------------------------------------------------------------------------

def _home_hash(home: Path) -> str:
    canonical = os.path.normcase(str(Path(home).expanduser().resolve(strict=False)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def windows_pipe_name(home: Path) -> str:
    """Per-HERMES_HOME named pipe path (Windows transport)."""
    return rf"\\.\pipe\hermes-gateway-{_home_hash(home)}"


def _pointer_path(home: Path) -> Path:
    return Path(home) / _POINTER_FILENAME


def _default_socket_path(home: Path) -> Path:
    return Path(home) / _SOCKET_FILENAME


def _fallback_socket_path(home: Path) -> Path:
    """Short temp-dir path for homes whose direct socket path exceeds sun_path.

    Prefers ``tempfile.gettempdir()``; when even that yields a too-long path
    (deep $TMPDIR), falls back to ``/tmp`` on POSIX. If nothing fits, the
    tempdir candidate is returned anyway — bind will fail non-fatally and
    consumers use the scan layer.
    """
    name = f"hermes-gw-{_home_hash(home)}.sock"
    candidates = [Path(tempfile.gettempdir()) / name]
    if not _IS_WINDOWS:
        candidates.append(Path("/tmp") / name)
    for candidate in candidates:
        if len(str(candidate).encode("utf-8")) <= _MAX_UNIX_PATH:
            return candidate
    return candidates[0]


def resolve_server_socket_path(home: Path) -> tuple[Path, Optional[Path]]:
    """Where the server should bind, plus the pointer file to write (or None).

    Returns ``(bind_path, pointer_file)``. ``pointer_file`` is non-None only
    when the direct in-home path is too long and the temp-dir fallback is in
    use — the server must then persist the real location for clients.
    """
    direct = _default_socket_path(home)
    if len(str(direct).encode("utf-8")) <= _MAX_UNIX_PATH:
        return direct, None
    return _fallback_socket_path(home), _pointer_path(home)


def resolve_client_socket_path(home: Path) -> Optional[Path]:
    """Where a client should connect for ``home``, or None when nothing exists."""
    direct = _default_socket_path(home)
    if direct.exists():
        return direct
    pointer = _pointer_path(home)
    try:
        if pointer.is_file():
            target = pointer.read_text(encoding="utf-8").strip()
            if target:
                candidate = Path(target)
                if candidate.exists():
                    return candidate
    except OSError:
        pass
    return None


# ---------------------------------------------------------------------------
# Default payload builders (import-light; overridable at wiring time)
# ---------------------------------------------------------------------------

def _detect_supervisor() -> str:
    """Best-effort supervisor kind for THIS process, from its own environment.

    Unlike the outside-in `_detect_supervisor_for_pid` scan, this answers from
    the process's own launch context — which is exactly the provenance the
    #92091 design wants declared rather than inferred.
    """
    env = os.environ
    if env.get("INVOCATION_ID"):
        return "systemd"
    if sys.platform == "darwin" and (
        env.get("XPC_SERVICE_NAME", "").startswith("ai.hermes")
        or env.get("LAUNCHD_SOCKET")
    ):
        return "launchd"
    if env.get("HERMES_DESKTOP_MANAGED"):
        return "desktop"
    if "--external-supervisor" in sys.argv:
        return "external"
    return "manual"


def build_identify_payload() -> dict[str, Any]:
    """Default ``identify`` answer, built from gateway.status primitives."""
    from gateway.status import (
        _build_pid_record,
        _get_code_identity_fields,
        _profile_label_for_home,
        read_runtime_status,
    )

    record = _build_pid_record()
    payload: dict[str, Any] = {
        "protocol": CONTROL_PROTOCOL_VERSION,
        "kind": record.get("kind"),
        "pid": record.get("pid"),
        "start_time": record.get("start_time"),
        "hermes_home": record.get("hermes_home"),
        "profile": _profile_label_for_home(record.get("hermes_home") or ""),
        "supervisor": _detect_supervisor(),
    }
    payload.update(_get_code_identity_fields())
    # served_profiles (multiplex mode) is stamped into the runtime status by
    # the runner; surface it when present so fleet consumers see coverage.
    try:
        runtime = read_runtime_status() or {}
        served = runtime.get("served_profiles")
        if isinstance(served, list) and served:
            payload["served_profiles"] = served
    except Exception:
        pass
    return payload


def build_status_payload() -> dict[str, Any]:
    """Default ``status`` answer — current runtime status, answered live."""
    from gateway.status import read_runtime_status

    payload = read_runtime_status() or {}
    payload = dict(payload)
    payload["protocol"] = CONTROL_PROTOCOL_VERSION
    payload["answered_at"] = time.time()
    payload["answering_pid"] = os.getpid()
    return payload


# ---------------------------------------------------------------------------
# Verb failures
# ---------------------------------------------------------------------------

class VerbError(Exception):
    """A verb failure that answers with the framework's failure envelope.

    The v1 envelope reports whether the verb *dispatched*, which is the wrong
    signal for verbs that perform an action that can fail (``platform_send``):
    a caller reading the top-level ``ok`` would read a failed send as success.
    Rather than hoisting result keys into the frame, an action verb raises
    ``VerbError`` and the server answers the ordinary failure frame
    ``{"ok": false, "protocol": 1, "id": …, "error": "<code>: <detail>"}`` —
    one envelope for every failure on the socket, with the machine-readable
    part as the code prefix. Plain-dict handlers (``identify``, ``status``,
    ``pause-for-update``) are untouched.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail

    @property
    def wire_error(self) -> str:
        """The ``error`` string carried by the failure frame."""
        return f"{self.code}: {self.detail}"


def _invoke_verb_handler(
    handler: Callable[..., dict[str, Any]], request: dict[str, Any]
) -> Any:
    """Call ``handler``, passing the raw request only if it accepts one.

    v1 handlers are zero-argument (the verb carries no parameters).
    Parameterised verbs such as ``platform_send`` declare a single positional
    parameter and receive the decoded request dict.
    """
    try:
        parameters = inspect.signature(handler).parameters
    except (TypeError, ValueError):  # builtins / C callables
        return handler()
    for param in parameters.values():
        if param.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            return handler(request)
    return handler()


# ---------------------------------------------------------------------------
# platform_send handler
# ---------------------------------------------------------------------------

def _platform_send_timeout(explicit: Optional[float] = None) -> float:
    """Send budget: explicit argument, else ``$HERMES_PLATFORM_SEND_TIMEOUT``."""
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(PLATFORM_SEND_TIMEOUT_ENV)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            logger.debug("Ignoring malformed %s=%r", PLATFORM_SEND_TIMEOUT_ENV, raw)
    return PLATFORM_SEND_DEFAULT_TIMEOUT


def _is_rate_limited(detail: str) -> bool:
    """WeCom signals throttling as errcode 846607 inside the failure text."""
    return PLATFORM_SEND_RATE_LIMIT_ERRCODE in detail


def build_platform_send_handler(
    get_adapter: Callable[[str], Any],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    timeout: Optional[float] = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the ``platform_send`` verb handler.

    ``get_adapter(platform)`` returns the live adapter for a platform name, or
    None when the gateway does not serve it. ``loop`` is the gateway's event
    loop — the handler itself runs on the control socket's executor thread, so
    the adapter's ``send`` coroutine is marshalled back onto that loop with
    ``asyncio.run_coroutine_threadsafe`` and awaited synchronously (the same
    thread→loop bridge ``pause-for-update`` uses, but result-carrying).

    The text is delivered verbatim: no templating, no command interpretation.
    ``request_id`` is opaque — echoed back and logged, never validated.
    Exactly one INFO line is logged per call, carrying the platform, chat id,
    text LENGTH, request id and outcome — never the message body.

    Success returns the contract's four-field result; every failure raises
    :class:`VerbError`, which the server turns into ``"<code>: <detail>"``.
    """

    def _handler(request: dict[str, Any]) -> dict[str, Any]:
        platform = request.get("platform")
        chat_id = request.get("chat_id")
        text = request.get("text")
        raw_request_id = request.get("request_id")
        # Opaque correlation token: echoed as-is when it is a string, and as
        # the empty string otherwise (absent, null, wrong type). Never a
        # reason to reject the request — the contract does not shape it.
        request_id = raw_request_id if isinstance(raw_request_id, str) else ""

        def _bad(detail: str) -> VerbError:
            logger.info(
                "platform_send platform=%r chat_id=%r request_id=%s result=%s",
                platform if isinstance(platform, str) else type(platform).__name__,
                chat_id if isinstance(chat_id, str) else type(chat_id).__name__,
                request_id,
                PLATFORM_SEND_ERR_BAD_REQUEST,
            )
            return VerbError(PLATFORM_SEND_ERR_BAD_REQUEST, detail)

        if not isinstance(platform, str) or not platform.strip():
            raise _bad("platform is required")
        platform = platform.strip().lower()
        if platform not in PLATFORM_SEND_SUPPORTED_PLATFORMS:
            raise _bad(
                f"unsupported platform {platform!r} "
                f"(supported: {', '.join(sorted(PLATFORM_SEND_SUPPORTED_PLATFORMS))})"
            )
        if not isinstance(chat_id, str) or not chat_id.strip():
            raise _bad("chat_id is required")
        chat_id = chat_id.strip()
        if not isinstance(text, str) or not text.strip():
            raise _bad("text is required and must be non-empty")
        if len(text) > PLATFORM_SEND_MAX_TEXT_CHARS:
            raise _bad(
                f"text exceeds {PLATFORM_SEND_MAX_TEXT_CHARS} characters "
                f"(got {len(text)})"
            )

        def _log(outcome: str) -> None:
            logger.info(
                "platform_send platform=%s chat_id=%s text_len=%d "
                "request_id=%s result=%s",
                platform, chat_id, len(text), request_id, outcome,
            )

        def _fail(code: str, detail: str) -> VerbError:
            _log(code)
            return VerbError(code, detail)

        def _fail_send(detail: str) -> VerbError:
            """send_failed, unless the platform said it was throttling us."""
            if _is_rate_limited(detail):
                return _fail(PLATFORM_SEND_ERR_RATE_LIMITED, detail)
            return _fail(PLATFORM_SEND_ERR_SEND_FAILED, detail)

        try:
            adapter = get_adapter(platform)
        except Exception as exc:
            adapter = None
            logger.debug("platform_send adapter lookup failed: %s", exc)
        if adapter is None:
            raise _fail(
                PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
                f"no live adapter for platform {platform!r}",
            )
        if not bool(getattr(adapter, "is_connected", True)):
            raise _fail(
                PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
                f"adapter for {platform!r} is not connected",
            )

        target_loop = loop or getattr(adapter, "_loop", None)
        if target_loop is None or target_loop.is_closed():
            raise _fail(
                PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
                "gateway event loop is not running",
            )

        budget = _platform_send_timeout(timeout)
        try:
            future = asyncio.run_coroutine_threadsafe(
                adapter.send(chat_id, text), target_loop
            )
        except Exception as exc:
            raise _fail_send(f"{type(exc).__name__}: {exc}") from exc

        try:
            result = future.result(timeout=budget)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise _fail(
                PLATFORM_SEND_ERR_TIMEOUT,
                f"adapter send did not complete within {budget}s",
            ) from exc
        except Exception as exc:
            raise _fail_send(f"{type(exc).__name__}: {exc}") from exc

        if getattr(result, "success", True) is False:
            raise _fail_send(
                str(getattr(result, "error", None) or "adapter reported failure")
            )

        message_id = getattr(result, "message_id", None)
        _log("ok")
        return {
            "message_id": "" if message_id is None else str(message_id),
            "request_id": request_id,
            "platform": platform,
            "chat_id": chat_id,
        }

    return _handler


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class GatewayControlServer:
    """Gateway-owned control socket server (identify/status, v1).

    Lifecycle is owned by the gateway process: ``start()`` after the PID-file
    claim (the point where this process becomes the authoritative gateway for
    its HERMES_HOME), ``stop()`` on shutdown. All failures are non-fatal —
    the gateway never refuses to serve messaging because its control socket
    couldn't bind; consumers simply fall back to the scan layer.
    """

    def __init__(
        self,
        home: Optional[Path] = None,
        *,
        verb_handlers: Optional[dict[str, Callable[[], dict[str, Any]]]] = None,
    ) -> None:
        if home is None:
            from gateway.status import _get_process_hermes_home

            home = _get_process_hermes_home()
        self._home = Path(home)
        self._server: Optional[asyncio.AbstractServer] = None
        self._pipe_server: Any = None  # Windows proactor pipe server
        self._bind_path: Optional[Path] = None
        self._pointer_file: Optional[Path] = None
        self._handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "identify": build_identify_payload,
            "status": build_status_payload,
        }
        if verb_handlers:
            self._handlers.update(verb_handlers)

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> bool:
        """Bind and start serving. Returns True on success, False otherwise."""
        try:
            if _IS_WINDOWS:
                return await self._start_windows()
            return await self._start_posix()
        except Exception as exc:
            logger.warning("Gateway control socket failed to start (non-fatal): %s", exc)
            return False

    async def _start_posix(self) -> bool:
        bind_path, pointer_file = resolve_server_socket_path(self._home)
        # Clear a stale socket left by a crashed predecessor. We only get
        # here after winning the PID-file O_EXCL race, so any existing file
        # is either stale or a plain collision — never a live sibling for
        # this HERMES_HOME.
        with contextlib.suppress(OSError):
            if bind_path.exists():
                bind_path.unlink()
        # Bind under a restrictive umask so the socket is never
        # world-connectable, even for the instant before an explicit chmod
        # could run. Restore the process umask immediately after.
        old_umask = os.umask(0o177)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_connection, path=str(bind_path)
            )
        finally:
            os.umask(old_umask)
        with contextlib.suppress(OSError):
            os.chmod(bind_path, 0o600)
        self._bind_path = bind_path
        if pointer_file is not None:
            pointer_file.write_text(str(bind_path), encoding="utf-8")
            self._pointer_file = pointer_file
        logger.info("Gateway control socket listening at %s", bind_path)
        return True

    async def _start_windows(self) -> bool:
        loop = asyncio.get_running_loop()
        start_serving_pipe = getattr(loop, "start_serving_pipe", None)
        if start_serving_pipe is None:
            logger.debug(
                "Event loop %s has no start_serving_pipe — control socket "
                "disabled (selector loop on Windows).",
                type(loop).__name__,
            )
            return False
        pipe_name = windows_pipe_name(self._home)

        def _factory():
            return _PipeControlProtocol(self)

        servers = await start_serving_pipe(_factory, pipe_name)
        self._pipe_server = servers[0] if servers else None
        logger.info("Gateway control pipe listening at %s", pipe_name)
        return self._pipe_server is not None

    async def stop(self) -> None:
        """Stop serving and remove the socket/pointer files."""
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        if self._pipe_server is not None:
            with contextlib.suppress(Exception):
                self._pipe_server.close()
            self._pipe_server = None
        self.cleanup_files()

    def cleanup_files(self) -> None:
        """Best-effort removal of socket + pointer files (atexit-safe)."""
        if self._bind_path is not None:
            with contextlib.suppress(OSError):
                self._bind_path.unlink(missing_ok=True)
        if self._pointer_file is not None:
            with contextlib.suppress(OSError):
                self._pointer_file.unlink(missing_ok=True)

    # -- request handling ----------------------------------------------------

    def handle_request_line(self, raw: bytes) -> bytes:
        """Process one JSON request line, return one JSON response line.

        Shared by the POSIX stream handler and the Windows pipe protocol.
        Never raises.
        """
        request_id: Any = None
        try:
            request = json.loads(raw.decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("request must be a JSON object")
            request_id = request.get("id")
            verb = request.get("verb")
            handler = self._handlers.get(verb) if isinstance(verb, str) else None
            if handler is None:
                response: dict[str, Any] = {
                    "ok": False,
                    "error": f"unknown verb: {verb!r}",
                    "protocol": CONTROL_PROTOCOL_VERSION,
                    "supported_verbs": sorted(self._handlers),
                }
            else:
                result = _invoke_verb_handler(handler, request)
                response = {
                    "ok": True,
                    "protocol": CONTROL_PROTOCOL_VERSION,
                    "result": result,
                }
        except VerbError as exc:
            # An action verb failed (see VerbError): the ordinary failure
            # envelope, with the machine-readable code as the error prefix.
            response = {
                "ok": False,
                "error": exc.wire_error,
                "protocol": CONTROL_PROTOCOL_VERSION,
            }
        except Exception as exc:
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "protocol": CONTROL_PROTOCOL_VERSION,
            }
        if request_id is not None:
            response["id"] = request_id
        try:
            encoded = json.dumps(response, default=str).encode("utf-8")
        except Exception:
            encoded = b'{"ok": false, "error": "response serialization failed"}'
        if len(encoded) > _MAX_RESPONSE_BYTES:
            encoded = b'{"ok": false, "error": "response too large"}'
        return encoded + b"\n"

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            raw = await asyncio.wait_for(
                reader.readline(), timeout=_DEFAULT_CLIENT_TIMEOUT
            )
            if not raw or len(raw) > _MAX_REQUEST_BYTES:
                return
            # Handlers read state files from disk; keep that off the
            # gateway's event loop (the same loop drives every platform
            # adapter), so a fast-polling consumer can't stall heartbeats.
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, self.handle_request_line, raw.rstrip(b"\n")
            )
            writer.write(response)
            await writer.drain()
        except (asyncio.TimeoutError, ConnectionError, OSError):
            pass
        except Exception:
            logger.debug("Control socket connection handler error", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                writer.close()


class _PipeControlProtocol(asyncio.Protocol):
    """One-shot request/response protocol for the Windows named pipe."""

    def __init__(self, server: GatewayControlServer) -> None:
        self._server = server
        self._transport: Any = None
        self._buffer = bytearray()

    def connection_made(self, transport) -> None:  # pragma: no cover - windows
        self._transport = transport

    def data_received(self, data: bytes) -> None:  # pragma: no cover - windows
        self._buffer.extend(data)
        if len(self._buffer) > _MAX_REQUEST_BYTES:
            self._transport.close()
            return
        if b"\n" in self._buffer:
            line, _, _ = bytes(self._buffer).partition(b"\n")
            try:
                self._transport.write(self._server.handle_request_line(line))
            finally:
                self._transport.close()


# ---------------------------------------------------------------------------
# Client (synchronous — used by CLI/updater consumers)
# ---------------------------------------------------------------------------

def query_gateway_control(
    home: Path,
    verb: str,
    *,
    timeout: float = _DEFAULT_CLIENT_TIMEOUT,
) -> Optional[dict[str, Any]]:
    """Ask the gateway serving ``home`` a control verb; None when unanswered.

    Returns the verb's ``result`` payload on success. Any failure — no
    socket, stale socket nobody accepts on, timeout, malformed answer,
    ``ok: false`` — returns None so callers fall back to the scan layer.
    Never raises.
    """
    request = (
        json.dumps({"verb": verb, "id": 1, "protocol": CONTROL_PROTOCOL_VERSION})
        .encode("utf-8")
        + b"\n"
    )
    try:
        if _IS_WINDOWS:
            raw = _query_windows_pipe(Path(home), request, timeout)
        else:
            raw = _query_unix_socket(Path(home), request, timeout)
    except Exception:
        return None
    if not raw:
        return None
    try:
        response = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(response, dict) or response.get("ok") is not True:
        return None
    result = response.get("result")
    return result if isinstance(result, dict) else None


def _query_unix_socket(home: Path, request: bytes, timeout: float) -> Optional[bytes]:
    path = resolve_client_socket_path(home)
    if path is None:
        return None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(str(path))
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            return None
        sock.sendall(request)
        chunks: list[bytes] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(65536)
            except socket.timeout:
                return None
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
            if sum(len(c) for c in chunks) > _MAX_RESPONSE_BYTES:
                return None
        data = b"".join(chunks)
        line, _, _ = data.partition(b"\n")
        return line or None


def _query_windows_pipe(
    home: Path, request: bytes, timeout: float
) -> Optional[bytes]:  # pragma: no cover - exercised on the wine2e lane
    pipe_name = windows_pipe_name(home)
    deadline = time.monotonic() + timeout
    handle = None
    while handle is None:
        try:
            handle = open(pipe_name, "r+b", buffering=0)
        except FileNotFoundError:
            return None
        except OSError:
            # Pipe busy (another client mid-handshake) — brief retry window.
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)
    try:
        handle.write(request)
        chunks: list[bytes] = []
        while time.monotonic() < deadline:
            chunk = handle.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
            if sum(len(c) for c in chunks) > _MAX_RESPONSE_BYTES:
                return None
        data = b"".join(chunks)
        line, _, _ = data.partition(b"\n")
        return line or None
    finally:
        with contextlib.suppress(Exception):
            handle.close()


def identify_gateway(home: Path, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT) -> Optional[dict[str, Any]]:
    """Convenience wrapper: ``identify`` the gateway serving ``home``."""
    return query_gateway_control(home, "identify", timeout=timeout)


def pause_gateway_for_update(
    home: Path, *, timeout: float = _DEFAULT_CLIENT_TIMEOUT
) -> Optional[dict[str, Any]]:
    """Ask the gateway serving ``home`` to drain and exit for an update.

    Step 2 of the socket migration (#92091). Returns the gateway's ACK —
    ``{"pausing": bool, "already_stopping": bool, "pid": int,
    "drain_timeout": float}`` — or None when no gateway answers (older
    gateway without the verb, no socket, dead socket). None means the
    caller falls back to the legacy pause path (signals / tree-kill),
    exactly as before this verb existed.
    """
    return query_gateway_control(home, "pause-for-update", timeout=timeout)


def _platform_send_client_failure(code: str, detail: str) -> dict[str, Any]:
    """Client-side failure shaped like the gateway's own failure answer."""
    return {"ok": False, "error": f"{code}: {detail}"}


def _normalised_error(error: str) -> str:
    """Keep a coded gateway error as-is; label an uncoded one.

    A gateway old enough to lack the verb answers ``"unknown verb: …"`` — no
    code prefix — so callers matching on the prefix would see nothing they
    recognise. Such answers are labelled ``platform_unavailable``, which is
    exactly what an unusable gateway means to the caller.
    """
    code, sep, _ = error.partition(": ")
    if sep and code in PLATFORM_SEND_ERR_CODES:
        return error
    return f"{PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE}: {error}"


def platform_send(
    home: Path,
    platform: str,
    chat_id: str,
    text: str,
    *,
    request_id: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict[str, Any]:
    """Ask the gateway serving ``home`` to send ``text`` to ``chat_id``.

    Returns the parsed answer envelope — ``{"ok": True, "result": {...}}`` on
    success (the result carrying ``message_id``/``request_id``/``platform``/
    ``chat_id``) or ``{"ok": False, "error": "<code>: <detail>"}``. Unlike
    :func:`query_gateway_control` this never collapses a failure to None: an
    unreachable gateway is itself reported as ``platform_unavailable``.

    ``request_id`` is an opaque correlation token echoed back in the result.

    The client budget is the gateway's send budget plus a small margin, so a
    gateway-side ``timeout`` answer arrives instead of the socket read dying
    first.
    """
    budget = _platform_send_timeout(timeout)
    payload: dict[str, Any] = {
        "verb": PLATFORM_SEND_VERB,
        "id": 1,
        "protocol": CONTROL_PROTOCOL_VERSION,
        "platform": platform,
        "chat_id": chat_id,
        "text": text,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    request = json.dumps(payload).encode("utf-8") + b"\n"
    try:
        if _IS_WINDOWS:
            raw = _query_windows_pipe(Path(home), request, budget + 5.0)
        else:
            raw = _query_unix_socket(Path(home), request, budget + 5.0)
    except Exception as exc:
        return _platform_send_client_failure(
            PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE, f"{type(exc).__name__}: {exc}"
        )
    if not raw:
        return _platform_send_client_failure(
            PLATFORM_SEND_ERR_PLATFORM_UNAVAILABLE,
            "no gateway answered the control socket",
        )
    try:
        response = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return _platform_send_client_failure(
            PLATFORM_SEND_ERR_SEND_FAILED, f"malformed control answer: {exc}"
        )
    if not isinstance(response, dict):
        return _platform_send_client_failure(
            PLATFORM_SEND_ERR_SEND_FAILED, "malformed control answer"
        )
    if response.get("ok") is True:
        result = response.get("result")
        if not isinstance(result, dict):
            return _platform_send_client_failure(
                PLATFORM_SEND_ERR_SEND_FAILED, "control answer carried no result"
            )
        return {"ok": True, "result": dict(result)}
    error = str(
        response.get("error") or "gateway does not support platform_send"
    )
    return {"ok": False, "error": _normalised_error(error)}
