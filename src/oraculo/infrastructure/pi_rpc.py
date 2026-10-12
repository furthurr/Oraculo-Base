"""Cliente RPC para comunicarse con Pi mediante JSONL sobre streams.

Implementa el puerto `PiAgentPort` del dominio. La comunicación usa el
protocolo RPC de Pi: registros JSON separados por LF en stdin/stdout.

Pi se ejecuta como subprocesso local (`pi --mode rpc`) con el directorio
de trabajo configurado. No requiere Docker.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable
from typing import Any

from oraculo.domain.pi import PiEvent, PiModeError, PiToolEvent

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 60.0
# Lectura por bloques: el framing JSONL de Pi no tiene límite de tamaño por registro.
READ_CHUNK_BYTES = 64 * 1024
# Protección de memoria, no límite funcional: un registro sin LF mayor se descarta.
MAX_RECORD_BYTES = 32 * 1024 * 1024
PROCESS_EXIT_TIMEOUT_S = 5.0


def _image_mime_type(data: bytes) -> str:
    """Identifica los formatos de imagen que el protocolo RPC de Pi admite."""
    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    raise PiModeError(
        "unsupported_image",
        user_message="Este modo admite imágenes JPEG, PNG, GIF o WebP.",
    )


class PiRpcClient:
    """Cliente async que implementa `PiAgentPort` sobre streams JSONL."""

    def __init__(self, *, reader: Any, writer: Any) -> None:
        self._reader = reader
        self._writer = writer
        self._event_callbacks: list[Callable[[PiEvent], None]] = []
        self._tool_callbacks: list[Callable[[PiToolEvent], None]] = []
        self._closed = False
        self._read_task: asyncio.Task[None] | None = None
        self._process: asyncio.subprocess.Process | None = None
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._request_seq = 0
        self._start_reading()

    @classmethod
    async def spawn(cls, *, cwd: str, session_file: str | None = None) -> PiRpcClient:
        """Lanza `pi --mode rpc` como subprocesso local y devuelve el cliente.

        `session_file` retoma una sesión guardada de Pi (`--session`). Se pasa
        como argumento separado, sin shell; su valor nunca se registra.
        """
        args = ["pi", "--mode", "rpc"]
        if session_file:
            args += ["--session", session_file]
        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except FileNotFoundError:
            raise PiModeError(
                "sandbox_unavailable",
                user_message="No se encontró el ejecutable necesario para el modo agente o no está en el PATH.",
            ) from None
        if process.stdin is None or process.stdout is None:
            raise PiModeError("process_crashed")
        client = cls(reader=process.stdout, writer=process.stdin)
        client._process = process
        asyncio.ensure_future(cls._drain_stderr(process))
        return client

    @staticmethod
    async def _drain_stderr(process: asyncio.subprocess.Process) -> None:
        """Lee stderr de Pi para registrar errores sin bloquear."""
        if process.stderr is None:
            return
        try:
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    logger.warning("Pi stderr: %s", text)
        except asyncio.CancelledError:
            pass

    def _start_reading(self) -> None:
        self._read_task = asyncio.ensure_future(self._read_loop())

    async def _read_loop(self) -> None:
        """Lee stdout por bloques y separa registros JSONL solo por LF (Req 1.1, 1.2).

        No usa `readline()`: `StreamReader` limita cada línea a 64 KiB y un
        registro mayor tumbaba el lector (incidente 2026-09-30).
        """
        logger.info("Pi RPC: read_loop iniciado")
        buffer = bytearray()
        discarding = False
        try:
            while not self._closed:
                chunk = await self._reader.read(READ_CHUNK_BYTES)
                if not chunk:
                    if buffer or discarding:
                        logger.warning("Pi RPC: registro final sin LF descartado")
                    logger.info("Pi RPC: stream terminado (sin más datos)")
                    break
                buffer += chunk
                while True:
                    newline = buffer.find(b"\n")
                    if newline < 0:
                        break
                    raw = bytes(buffer[:newline])
                    del buffer[: newline + 1]
                    if discarding:
                        # Cola de un registro sobredimensionado: se descarta.
                        discarding = False
                        continue
                    self._handle_raw_record(raw)
                if len(buffer) > MAX_RECORD_BYTES:
                    logger.warning(
                        "Pi RPC: registro de más de %d bytes sin LF descartado",
                        MAX_RECORD_BYTES,
                    )
                    buffer.clear()
                    discarding = True
        except asyncio.CancelledError:
            logger.info("Pi RPC: read_loop cancelado")
        except Exception as exc:
            logger.warning("Pi RPC: error en read_loop: %s", type(exc).__name__)
        finally:
            await self._on_reader_finished()

    def _handle_raw_record(self, raw: bytes) -> None:
        """Decodifica un registro; nunca registra su contenido (Req 1.3)."""
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        if not raw.strip():
            return
        try:
            record = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            logger.warning("Pi RPC: JSONL inválido ignorado bytes=%d", len(raw))
            return
        if not isinstance(record, dict):
            logger.warning("Pi RPC: registro no es un objeto JSON; ignorado")
            return
        logger.debug(
            "Pi RPC: recibido tipo=%s bytes=%d", record.get("type", ""), len(raw)
        )
        self._dispatch(record)

    async def _on_reader_finished(self) -> None:
        """Si la caída no fue pedida, cierra el cliente, avisa y termina Pi (Req 2.1, 2.2)."""
        unexpected = not self._closed
        self._closed = True
        self._fail_pending(PiModeError("process_crashed"))
        if not unexpected:
            return
        logger.warning("Pi RPC: el lector terminó de forma inesperada; se cierra la sesión")
        self._emit_event(PiEvent(type="closed", content=""))
        await self._terminate_process()

    async def _terminate_process(self) -> None:
        """Termina el proceso de Pi: `terminate` y, si no sale a tiempo, `kill`."""
        process = self._process
        if process is None or process.returncode is not None:
            return
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=PROCESS_EXIT_TIMEOUT_S)
        except ProcessLookupError:
            return
        except asyncio.TimeoutError:
            try:
                process.kill()
            except ProcessLookupError:
                logger.info("Pi RPC: el proceso ya había terminado")

    def _fail_pending(self, error: PiModeError) -> None:
        """Falla todas las peticiones pendientes (fin de stream o cierre)."""
        for request_id, future in list(self._pending.items()):
            self._pending.pop(request_id, None)
            if not future.done():
                future.set_exception(error)

    def _dispatch(self, record: dict[str, Any]) -> None:
        record_type = record.get("type", "")

        if record_type == "response":
            self._handle_response(record)
        elif record_type == "message_update":
            self._handle_message_update(record)
        elif record_type == "message_end":
            self._handle_message_end(record)
        elif record_type == "tool_execution_start":
            self._handle_tool_event("tool_start", record)
        elif record_type == "tool_execution_update":
            self._handle_tool_event("tool_update", record)
        elif record_type == "tool_execution_end":
            self._handle_tool_event("tool_end", record)
        elif record_type == "agent_settled":
            # Pi no seguirá trabajando de forma automática (Req 3.1).
            self._emit_event(PiEvent(type="settled", content=""))

    def _handle_response(self, record: dict[str, Any]) -> None:
        request_id = record.get("id")
        if not isinstance(request_id, str):
            return
        future = self._pending.pop(request_id, None)
        if future is None or future.done():
            return
        future.set_result(record)

    def _handle_message_update(self, record: dict[str, Any]) -> None:
        event_data = record.get("assistantMessageEvent", {})
        event_type = event_data.get("type", "")

        if event_type == "text_delta":
            delta = event_data.get("delta", "")
            if delta:
                self._emit_event(PiEvent(type="text_delta", content=delta))
        elif event_type == "error":
            reason = event_data.get("reason", "unknown")
            error_msg = event_data.get("error", "")
            self._emit_event(PiEvent(type="error", content=f"{reason}: {error_msg}"))
        elif event_type:
            # Razonamiento u otros bloques: solo actividad para el timeout (Req 3.3).
            self._emit_event(PiEvent(type="activity", content=""))

    def _handle_message_end(self, record: dict[str, Any]) -> None:
        message = record.get("message", {})
        role = message.get("role", "")
        if role != "assistant":
            return
        content = message.get("content", "")
        if isinstance(content, str) and content:
            self._emit_event(PiEvent(type="message_end", content=content))
        elif isinstance(content, list):
            text_parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            combined = "".join(text_parts)
            if combined:
                self._emit_event(PiEvent(type="message_end", content=combined))

    def _handle_tool_event(self, event_type: str, record: dict[str, Any]) -> None:
        tool_name = record.get("toolName", "")
        event = PiToolEvent(type=event_type, tool_name=tool_name)
        for callback in self._tool_callbacks:
            try:
                callback(event)
            except Exception:
                logger.warning("Callback de herramienta falló")

    def _emit_event(self, event: PiEvent) -> None:
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception:
                logger.warning("Callback de evento falló")

    async def _send_command(self, command: dict[str, Any]) -> None:
        if self._closed:
            raise PiModeError("process_crashed")
        line = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
        self._writer.write((line + "\n").encode("utf-8"))
        await self._writer.drain()

    async def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        """Envía un comando y espera su `response` correlacionado por `id`."""
        if self._closed:
            raise PiModeError("process_crashed")
        self._request_seq += 1
        request_id = f"req-{self._request_seq}"
        future: asyncio.Future[dict[str, Any]] = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        await self._send_command({**command, "id": request_id})
        try:
            response = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise PiModeError(
                "rpc_timeout",
                user_message="El agente no respondió a tiempo.",
            ) from None
        if not response.get("success", False):
            error = response.get("error", "error desconocido")
            raise PiModeError(
                "rpc_error",
                user_message=f"El agente no pudo completar la operación: {error}",
            )
        data = response.get("data")
        return data if isinstance(data, dict) else {}

    async def send_prompt(
        self, text: str, images: list[bytes] | None = None
    ) -> None:
        command: dict[str, Any] = {
            "id": f"prompt-{id(text)}",
            "type": "prompt",
            "message": text,
        }
        if images:
            command["images"] = [
                {
                    "type": "image",
                    "data": base64.b64encode(image).decode("ascii"),
                    "mimeType": _image_mime_type(image),
                }
                for image in images
            ]
        await self._send_command(command)

    async def send_steer(
        self, text: str, images: list[bytes] | None = None
    ) -> None:
        command: dict[str, Any] = {
            "id": f"steer-{id(text)}",
            "type": "steer",
            "message": text,
        }
        if images:
            command["images"] = [
                {
                    "type": "image",
                    "data": base64.b64encode(image).decode("ascii"),
                    "mimeType": _image_mime_type(image),
                }
                for image in images
            ]
        await self._send_command(command)

    async def abort(self) -> None:
        await self._send_command({"id": "abort-1", "type": "abort"})

    async def new_session(self) -> None:
        await self._send_command({"id": "new-session-1", "type": "new_session"})

    async def get_state(self) -> dict[str, Any]:
        return await self._request({"type": "get_state"})

    async def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        return await self._request({
            "type": "set_model",
            "provider": provider,
            "modelId": model_id,
        })

    async def get_available_models(self) -> list[dict[str, Any]]:
        data = await self._request({"type": "get_available_models"})
        models = data.get("models", [])
        return [m for m in models if isinstance(m, dict)] if isinstance(models, list) else []

    async def get_commands(self) -> list[dict[str, Any]]:
        data = await self._request({"type": "get_commands"})
        commands = data.get("commands", [])
        return [c for c in commands if isinstance(c, dict)] if isinstance(commands, list) else []

    async def bash(self, command: str) -> dict[str, Any]:
        return await self._request({"type": "bash", "command": command})

    def on_event(self, callback: Callable[[PiEvent], None]) -> None:
        self._event_callbacks.append(callback)

    def on_tool_event(self, callback: Callable[[PiToolEvent], None]) -> None:
        self._tool_callbacks.append(callback)

    async def close(self) -> None:
        self._closed = True
        self._fail_pending(PiModeError("process_crashed"))
        if self._read_task is not None:
            self._read_task.cancel()
            try:
                await self._read_task
            except asyncio.CancelledError:
                pass
        await self._terminate_process()

    @property
    def closed(self) -> bool:
        return self._closed