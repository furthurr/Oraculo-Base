"""Tests del cliente RPC de Pi: serialización JSONL y manejo de eventos."""

from __future__ import annotations

import asyncio
import base64
import json

import pytest

from oraculo.domain.pi import PiEvent, PiModeError, PiToolEvent
from oraculo.infrastructure.pi_rpc import PiRpcClient


class FakeStream:
    """Stream simulado para tests: cola de entrada y registro de salida."""

    def __init__(self) -> None:
        self._input: asyncio.Queue[bytes] = asyncio.Queue()
        self.sent: list[str] = []

    def feed(self, line: str) -> None:
        self._input.put_nowait((line + "\n").encode("utf-8"))

    async def close(self) -> None:
        self._input.put_nowait(b"")

    async def read(self, n: int = -1) -> bytes:
        # El lector consume por bloques; cada `feed` es un bloque (b"" = EOF).
        del n
        return await self._input.get()

    def write(self, data: bytes) -> None:
        self.sent.append(data.decode("utf-8"))

    async def drain(self) -> None:
        pass


def make_client() -> tuple[PiRpcClient, FakeStream, FakeStream]:
    reader = FakeStream()
    writer = FakeStream()
    client = PiRpcClient(reader=reader, writer=writer)  # type: ignore[arg-type]
    return client, reader, writer


class TestSendPrompt:
    async def test_prompt_serializes_to_jsonl(self) -> None:
        client, _, writer = make_client()
        await client.send_prompt("hola")
        assert len(writer.sent) == 1
        record = json.loads(writer.sent[0])
        assert record["type"] == "prompt"
        assert record["message"] == "hola"

    async def test_prompt_includes_id(self) -> None:
        client, _, writer = make_client()
        await client.send_prompt("hola")
        record = json.loads(writer.sent[0])
        assert "id" in record

    async def test_prompt_with_images(self) -> None:
        client, _, writer = make_client()
        image = b"\x89PNG\r\n\x1a\ncontenido"
        await client.send_prompt("mira", images=[image])
        record = json.loads(writer.sent[0])
        assert record["type"] == "prompt"
        assert record["images"][0]["type"] == "image"
        assert record["images"][0]["mimeType"] == "image/png"
        assert record["images"][0]["data"] == base64.b64encode(image).decode("ascii")

    async def test_jpeg_uses_its_actual_mime_type(self) -> None:
        client, _, writer = make_client()
        await client.send_prompt("mira", images=[b"\xff\xd8\xffjpeg"])
        record = json.loads(writer.sent[0])
        assert record["images"][0]["mimeType"] == "image/jpeg"

    async def test_unsupported_image_is_rejected(self) -> None:
        client, _, writer = make_client()
        with pytest.raises(PiModeError, match="unsupported_image"):
            await client.send_prompt("mira", images=[b"no es imagen"])
        assert writer.sent == []


class TestSendSteer:
    async def test_steer_serializes_to_jsonl(self) -> None:
        client, _, writer = make_client()
        await client.send_steer("cambia el rumbo")
        record = json.loads(writer.sent[0])
        assert record["type"] == "steer"
        assert record["message"] == "cambia el rumbo"

    async def test_steer_can_include_images(self) -> None:
        client, _, writer = make_client()
        await client.send_steer("mira", images=[b"\xff\xd8\xffjpeg"])
        record = json.loads(writer.sent[0])
        assert record["type"] == "steer"
        assert record["images"][0]["mimeType"] == "image/jpeg"


class TestAbort:
    async def test_abort_serializes_to_jsonl(self) -> None:
        client, _, writer = make_client()
        await client.abort()
        record = json.loads(writer.sent[0])
        assert record["type"] == "abort"


class TestNewSession:
    async def test_new_session_serializes_to_jsonl(self) -> None:
        client, _, writer = make_client()
        await client.new_session()
        record = json.loads(writer.sent[0])
        assert record["type"] == "new_session"


class TestEventHandling:
    async def test_text_delta_emits_event(self) -> None:
        client, reader, _ = make_client()
        events: list[PiEvent] = []
        client.on_event(events.append)

        reader.feed(json.dumps({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "hola"},
        }))
        await asyncio.sleep(0.05)
        assert len(events) == 1
        assert events[0].type == "text_delta"
        assert events[0].content == "hola"

    async def test_message_end_emits_event(self) -> None:
        client, reader, _ = make_client()
        events: list[PiEvent] = []
        client.on_event(events.append)

        reader.feed(json.dumps({
            "type": "message_end",
            "message": {"role": "assistant", "content": "respuesta final"},
        }))
        await asyncio.sleep(0.05)
        assert len(events) == 1
        assert events[0].type == "message_end"
        assert events[0].content == "respuesta final"

    async def test_tool_start_emits_tool_event(self) -> None:
        client, reader, _ = make_client()
        tool_events: list[PiToolEvent] = []
        client.on_tool_event(tool_events.append)

        reader.feed(json.dumps({
            "type": "tool_execution_start",
            "toolCallId": "abc",
            "toolName": "read",
            "args": {},
        }))
        await asyncio.sleep(0.05)
        assert len(tool_events) == 1
        assert tool_events[0].type == "tool_start"
        assert tool_events[0].tool_name == "read"

    async def test_error_event_emitted(self) -> None:
        client, reader, _ = make_client()
        events: list[PiEvent] = []
        client.on_event(events.append)

        reader.feed(json.dumps({
            "type": "message_update",
            "assistantMessageEvent": {"type": "error", "reason": "model_error", "error": "fallo"},
        }))
        await asyncio.sleep(0.05)
        assert any(e.type == "error" for e in events)

    async def test_ignores_unknown_events(self) -> None:
        client, reader, _ = make_client()
        events: list[PiEvent] = []
        client.on_event(events.append)

        reader.feed(json.dumps({"type": "queue_update", "steering": [], "followUp": []}))
        await asyncio.sleep(0.05)
        assert len(events) == 0


class TestRequestResponse:
    async def test_get_state_waits_for_response(self) -> None:
        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.get_state())
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"isStreaming": False, "thinkingLevel": "medium"},
        }))
        state = await asyncio.wait_for(task, timeout=1)
        assert state == {"isStreaming": False, "thinkingLevel": "medium"}

    async def test_response_error_raises_pi_mode_error(self) -> None:
        from oraculo.domain.pi import PiModeError

        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.get_state())
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "get_state",
            "success": False,
            "error": "fallo interno",
        }))
        with pytest.raises(PiModeError):
            await asyncio.wait_for(task, timeout=1)

    async def test_correlates_by_request_id(self) -> None:
        client, reader, writer = make_client()
        first = asyncio.ensure_future(client.get_state())
        await asyncio.sleep(0.05)
        second = asyncio.ensure_future(client.get_state())
        await asyncio.sleep(0.05)
        ids = [json.loads(line)["id"] for line in writer.sent]
        # Respondemos en orden inverso: cada future debe recibir su data.
        reader.feed(json.dumps({
            "id": ids[1],
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"marker": "segundo"},
        }))
        reader.feed(json.dumps({
            "id": ids[0],
            "type": "response",
            "command": "get_state",
            "success": True,
            "data": {"marker": "primero"},
        }))
        assert (await asyncio.wait_for(first, timeout=1)) == {"marker": "primero"}
        assert (await asyncio.wait_for(second, timeout=1)) == {"marker": "segundo"}

    async def test_stream_end_fails_pending_request(self) -> None:
        from oraculo.domain.pi import PiModeError

        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.get_state())
        await asyncio.sleep(0.05)
        await reader.close()
        with pytest.raises(PiModeError):
            await asyncio.wait_for(task, timeout=1)


class TestModelAndToolsCommands:
    async def test_set_model_serializes_and_returns_model(self) -> None:
        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.set_model("anthropic", "claude-x"))
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        assert request["type"] == "set_model"
        assert request["provider"] == "anthropic"
        assert request["modelId"] == "claude-x"
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "set_model",
            "success": True,
            "data": {"id": "claude-x", "name": "Claude X", "provider": "anthropic"},
        }))
        model = await asyncio.wait_for(task, timeout=1)
        assert model["id"] == "claude-x"

    async def test_set_model_failure_raises(self) -> None:
        from oraculo.domain.pi import PiModeError

        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.set_model("bad", "nope"))
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "set_model",
            "success": False,
            "error": "Model not found: bad/nope",
        }))
        with pytest.raises(PiModeError):
            await asyncio.wait_for(task, timeout=1)

    async def test_get_available_models_returns_list(self) -> None:
        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.get_available_models())
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        assert request["type"] == "get_available_models"
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "get_available_models",
            "success": True,
            "data": {"models": [
                {"id": "m1", "name": "Modelo 1", "provider": "p"},
                {"id": "m2", "name": "Modelo 2", "provider": "p"},
            ]},
        }))
        models = await asyncio.wait_for(task, timeout=1)
        assert [m["id"] for m in models] == ["m1", "m2"]

    async def test_get_commands_returns_list(self) -> None:
        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.get_commands())
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        assert request["type"] == "get_commands"
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "get_commands",
            "success": True,
            "data": {"commands": [{"name": "fix-tests", "description": "Fix", "source": "prompt"}]},
        }))
        commands = await asyncio.wait_for(task, timeout=1)
        assert commands[0]["name"] == "fix-tests"

    async def test_bash_serializes_and_returns_result(self) -> None:
        client, reader, writer = make_client()
        task = asyncio.ensure_future(client.bash("ls -la"))
        await asyncio.sleep(0.05)
        request = json.loads(writer.sent[0])
        assert request["type"] == "bash"
        assert request["command"] == "ls -la"
        reader.feed(json.dumps({
            "id": request["id"],
            "type": "response",
            "command": "bash",
            "success": True,
            "data": {"output": "total 0", "exitCode": 0, "cancelled": False, "truncated": False},
        }))
        result = await asyncio.wait_for(task, timeout=1)
        assert result["output"] == "total 0"
        assert result["exitCode"] == 0


class TestClose:
    async def test_close_terminates(self) -> None:
        client, _, _ = make_client()
        await client.close()
        assert client.closed