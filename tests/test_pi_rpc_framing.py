"""Regresión del lector RPC de Pi: framing JSONL, cierre inesperado y eventos.

Spec: `.sdd/specs/modo-pi/bugfix-lector-rpc/` (Req 1.x, 2.1, 2.2, 2.3, 3.1).
"""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from oraculo.domain.pi import PiEvent, PiModeError, PiToolEvent
from oraculo.infrastructure import pi_rpc
from oraculo.infrastructure.pi_rpc import PiRpcClient


class NullWriter:
    """Writer que descarta lo escrito; los tests de lectura no lo necesitan."""

    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.sent.append(data)

    async def drain(self) -> None:
        return None


def make_reader_client() -> tuple[PiRpcClient, asyncio.StreamReader, list[PiEvent], list[PiToolEvent]]:
    """Cliente sobre un `asyncio.StreamReader` real (límite de 64 KiB por defecto)."""
    reader = asyncio.StreamReader()
    client = PiRpcClient(reader=reader, writer=NullWriter())
    events: list[PiEvent] = []
    tools: list[PiToolEvent] = []
    client.on_event(events.append)
    client.on_tool_event(tools.append)
    return client, reader, events, tools


def record(obj: dict[str, object]) -> bytes:
    return json.dumps(obj).encode("utf-8") + b"\n"


def text_end(text: str) -> dict[str, object]:
    return {"type": "message_end", "message": {"role": "assistant", "content": text}}


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.01)


class TestFraming:
    async def test_record_larger_than_64kib_is_dispatched(self) -> None:
        """Req 1.1 — incidente 2026-09-30 08:20 (`chunk is longer than limit`)."""
        client, reader, events, tools = make_reader_client()
        big = {
            "type": "tool_execution_update",
            "toolCallId": "c1",
            "toolName": "bash",
            "partialResult": {"content": [{"type": "text", "text": "x" * 200_000}]},
        }
        reader.feed_data(record(big))
        reader.feed_data(record(text_end("después")))
        await settle()
        assert not client.closed
        assert [e.content for e in events if e.type == "message_end"] == ["después"]
        await client.close()

    async def test_records_split_in_one_byte_chunks_with_crlf_and_utf8(self) -> None:
        """Req 1.2 — reconstrucción por LF, CR opcional y cortes dentro de UTF-8."""
        client, reader, events, _ = make_reader_client()
        data = (
            json.dumps(text_end("ñandú π")).encode("utf-8") + b"\r\n"
            + record(text_end("segundo"))
        )
        for i in range(len(data)):
            reader.feed_data(data[i : i + 1])
            await asyncio.sleep(0)
        await settle()
        assert [e.content for e in events if e.type == "message_end"] == ["ñandú π", "segundo"]
        await client.close()

    async def test_several_records_in_one_chunk(self) -> None:
        """Req 1.2."""
        client, reader, events, _ = make_reader_client()
        reader.feed_data(record(text_end("a")) + record(text_end("b")) + record(text_end("c")))
        await settle()
        assert [e.content for e in events] == ["a", "b", "c"]
        await client.close()

    async def test_invalid_json_is_skipped_without_logging_content(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 1.3."""
        client, reader, events, _ = make_reader_client()
        with caplog.at_level(logging.INFO, logger="oraculo.infrastructure.pi_rpc"):
            reader.feed_data(record(text_end("a")) + b"{no-json SECRETO\n" + record(text_end("b")))
            await settle()
        assert [e.content for e in events] == ["a", "b"]
        assert "SECRETO" not in caplog.text
        await client.close()

    async def test_oversized_record_without_lf_is_discarded_and_reading_continues(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Req 1.3 — protección de memoria: se descarta hasta el siguiente LF."""
        monkeypatch.setattr(pi_rpc, "MAX_RECORD_BYTES", 1024)
        client, reader, events, _ = make_reader_client()
        reader.feed_data(b"x" * 5000)
        await settle()
        reader.feed_data(b"yyy\n" + record(text_end("ok")))
        await settle()
        assert not client.closed
        assert [e.content for e in events] == ["ok"]
        await client.close()


class FakeProcess:
    """Proceso fake: registra terminate/kill; `wait` termina tras `terminate`."""

    def __init__(self, *, hangs: bool = False) -> None:
        self.terminated = False
        self.killed = False
        self._hangs = hangs
        self._done = asyncio.Event()
        self.returncode: int | None = None

    def terminate(self) -> None:
        self.terminated = True
        if not self._hangs:
            self.returncode = 0
            self._done.set()

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode or 0


class TestUnexpectedClose:
    async def test_eof_emits_closed_marks_client_and_fails_pending(self) -> None:
        """Req 2.1, 2.2."""
        client, reader, events, _ = make_reader_client()
        pending = asyncio.ensure_future(client.get_state())
        await settle()
        reader.feed_eof()
        await settle()
        assert client.closed
        assert [e.type for e in events] == ["closed"]
        with pytest.raises(PiModeError):
            await asyncio.wait_for(pending, timeout=1)

    async def test_reader_error_terminates_process(self) -> None:
        """Req 2.2 — el proceso no queda huérfano (invariante 4)."""
        client, reader, events, _ = make_reader_client()
        process = FakeProcess()
        client._process = process  # type: ignore[assignment]
        reader.set_exception(OSError("pipe roto"))
        await settle()
        assert client.closed
        assert process.terminated
        assert [e.type for e in events] == ["closed"]

    async def test_process_that_ignores_terminate_is_killed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Req 2.2."""
        monkeypatch.setattr(pi_rpc, "PROCESS_EXIT_TIMEOUT_S", 0.05)
        client, reader, _, _ = make_reader_client()
        process = FakeProcess(hangs=True)
        client._process = process  # type: ignore[assignment]
        reader.feed_eof()
        for _ in range(20):
            await asyncio.sleep(0.01)
        assert process.terminated and process.killed

    async def test_explicit_close_does_not_emit_closed_and_is_idempotent(self) -> None:
        """Req 2.2 — solo la caída no pedida avisa."""
        client, _, events, _ = make_reader_client()
        await client.close()
        await client.close()
        await settle()
        assert client.closed
        assert events == []


class TestNewEvents:
    async def test_agent_settled_emits_settled(self) -> None:
        """Req 3.1."""
        client, reader, events, _ = make_reader_client()
        reader.feed_data(record({"type": "agent_settled"}))
        await settle()
        assert [e.type for e in events] == ["settled"]
        await client.close()

    async def test_tool_execution_update_emits_tool_update(self) -> None:
        """Req 3.3 — actividad para el timeout."""
        client, reader, _, tools = make_reader_client()
        reader.feed_data(record({
            "type": "tool_execution_update",
            "toolCallId": "c1",
            "toolName": "bash",
            "partialResult": {"content": []},
        }))
        await settle()
        assert [(t.type, t.tool_name) for t in tools] == [("tool_update", "bash")]
        await client.close()

    async def test_thinking_delta_emits_activity_without_content(self) -> None:
        """Req 3.3 — el razonamiento no se muestra en este bugfix."""
        client, reader, events, _ = make_reader_client()
        reader.feed_data(record({
            "type": "message_update",
            "assistantMessageEvent": {"type": "thinking_delta", "contentIndex": 0, "delta": "pienso"},
        }))
        await settle()
        assert [(e.type, e.content) for e in events] == [("activity", "")]
        await client.close()


class TestSpawnSessionFile:
    async def _spawn_args(
        self, monkeypatch: pytest.MonkeyPatch, session_file: str | None
    ) -> tuple[object, ...]:
        captured: list[tuple[object, ...]] = []

        async def fake_exec(*args: object, **kwargs: object) -> object:
            captured.append(args)
            proc = FakeProcess()
            proc.stdin = NullWriter()  # type: ignore[attr-defined]
            proc.stdout = asyncio.StreamReader()  # type: ignore[attr-defined]
            proc.stderr = None  # type: ignore[attr-defined]
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        client = await PiRpcClient.spawn(cwd="/tmp", session_file=session_file)
        await client.close()
        return captured[0]

    async def test_spawn_passes_session_as_separate_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Req 2.3 — `--session` como argumento, sin shell."""
        args = await self._spawn_args(monkeypatch, "/ruta con espacios/s.jsonl")
        assert args == ("pi", "--mode", "rpc", "--session", "/ruta con espacios/s.jsonl")

    async def test_spawn_without_session_keeps_default_args(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        args = await self._spawn_args(monkeypatch, None)
        assert args == ("pi", "--mode", "rpc")
