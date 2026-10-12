"""Tests del envío de adjuntos a Codex/GPT (Wave 5, tarea 5.2)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import httpx
import pytest
import respx

from oraculo.domain.attachments import AttachmentMode, AttachmentPayload
from oraculo.domain.errors import AIError
from oraculo.domain.models import ChatMessage, MessageRole
from oraculo.infrastructure.codex import CodexClient
from oraculo.infrastructure.codex_auth import OAuthCredential, OAuthCredentialStore

URL = "https://chatgpt.com/backend-api/codex/responses"
MESSAGES = [ChatMessage(MessageRole.USER, "describe los archivos")]


def event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def finished(text: str = "Hecho") -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"},
        text="".join((
            event("response.output_text.delta", {"delta": text}),
            event("response.completed", {"response": {"status": "completed"}}),
        )),
    )


async def make_client(tmp_path: Path) -> CodexClient:
    store = OAuthCredentialStore(tmp_path / "private" / "auth.json")
    await store.save(OAuthCredential(
        "access-SENTINEL", "refresh-SENTINEL", 4102444800, "account-SENTINEL",
    ))
    return CodexClient(store=store, http_client=httpx.AsyncClient(),
                       system_prompt="BASE", timeout_s=5)


def text_payload(name: str, text: str) -> AttachmentPayload:
    return AttachmentPayload(
        name=name, mime_type="text/plain", mode=AttachmentMode.TEXT, text=text,
    )


def binary_payload(name: str, data: bytes, mime: str = "image/jpeg") -> AttachmentPayload:
    return AttachmentPayload(
        name=name, mime_type=mime, mode=AttachmentMode.BINARY, data=data,
    )


@respx.mock
async def test_texto_adjunto_se_envia_como_input_file(tmp_path: Path):
    respx.post(URL).mock(return_value=finished())
    client = await make_client(tmp_path)

    await client.complete(MESSAGES, attachments=(text_payload("nota.txt", "dato clave"),))

    body = json.loads(respx.calls[0].request.read())
    serialized = json.dumps(body, ensure_ascii=False)
    assert "nota.txt" in serialized
    assert "dato clave" in serialized


@respx.mock
async def test_binario_se_envia_como_file_data_base64(tmp_path: Path):
    respx.post(URL).mock(return_value=finished())
    client = await make_client(tmp_path)

    await client.complete(MESSAGES, attachments=(binary_payload("foto.jpg", b"\xff\xd8"),))

    body = json.loads(respx.calls[0].request.read())
    serialized = json.dumps(body)
    encoded = base64.b64encode(b"\xff\xd8").decode()
    assert encoded in serialized
    assert "file_data" in serialized or "input_image" in serialized


@respx.mock
async def test_imagen_usa_input_image(tmp_path: Path):
    respx.post(URL).mock(return_value=finished())
    client = await make_client(tmp_path)

    await client.complete(
        MESSAGES,
        attachments=(binary_payload("foto.jpg", b"\xff\xd8", mime="image/jpeg"),),
    )

    body = json.loads(respx.calls[0].request.read())
    image_block = body["input"][0]["content"][1]
    assert image_block == {
        "type": "input_image",
        "image_url": image_block["image_url"],
    }
    assert "data:image/jpeg;base64," in json.dumps(body)


@respx.mock
async def test_error_del_proveedor_se_propaga_como_ai_error(tmp_path: Path):
    respx.post(URL).mock(return_value=httpx.Response(
        400, json={"error": {"message": "archivo demasiado grande"}},
    ))
    client = await make_client(tmp_path)

    with pytest.raises(AIError):
        await client.complete(MESSAGES, attachments=(binary_payload("foto.jpg", b"\xff\xd8"),))


@respx.mock
async def test_sin_adjuntos_el_payload_no_cambia(tmp_path: Path):
    respx.post(URL).mock(return_value=finished())
    client = await make_client(tmp_path)

    await client.complete(MESSAGES)

    serialized = json.dumps(json.loads(respx.calls[0].request.read()))
    assert "file_data" not in serialized
    assert "input_image" not in serialized
