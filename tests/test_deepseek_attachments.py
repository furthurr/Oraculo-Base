"""Tests del envío de adjuntos a DeepSeek (Wave 5, tarea 5.1)."""

from __future__ import annotations

import httpx
import pytest
import respx

from oraculo.domain.attachments import (
    AttachmentMode,
    AttachmentPayload,
)
from oraculo.domain.errors import AIError
from oraculo.domain.models import ChatMessage, MessageRole
from oraculo.infrastructure.deepseek import DEEPSEEK_BASE_URL, DeepSeekClient

URL = f"{DEEPSEEK_BASE_URL}/chat/completions"
FILES_URL = f"{DEEPSEEK_BASE_URL}/files"


def make_client() -> DeepSeekClient:
    return DeepSeekClient(
        api_key="sk-test",
        model="deepseek-flash",
        timeout_s=5.0,
        max_tokens=128,
        system_prompt="Responde en español",
        thinking=False,
        reasoning_effort="low",
        http_client=httpx.AsyncClient(),
    )


def completion(content: str = "Hecho") -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
    )


def text_payload(name: str, text: str) -> AttachmentPayload:
    return AttachmentPayload(
        name=name, mime_type="text/plain", mode=AttachmentMode.TEXT, text=text,
    )


def binary_payload(name: str, data: bytes) -> AttachmentPayload:
    return AttachmentPayload(
        name=name, mime_type="image/jpeg", mode=AttachmentMode.BINARY, data=data,
    )


@respx.mock
async def test_texto_adjunto_viaja_en_el_contenido_del_mensaje():
    respx.post(URL).mock(return_value=completion())
    client = make_client()

    await client.complete(
        [ChatMessage(MessageRole.USER, "resume")],
        attachments=(text_payload("nota.txt", "contenido secreto"),),
    )

    request = respx.calls[0].request
    body = request.read().decode()
    assert "contenido secreto" in body
    assert "nota.txt" in body
    assert "resume" in body


@respx.mock
async def test_binario_se_sube_por_files_api_y_se_referencia_por_file_id():
    respx.post(FILES_URL).mock(return_value=httpx.Response(
        200, json={"id": "file-api-abc", "object": "file", "bytes": 2},
    ))
    respx.post(URL).mock(return_value=completion())
    client = make_client()

    await client.complete(
        [ChatMessage(MessageRole.USER, "describe")],
        attachments=(binary_payload("foto.jpg", b"\xff\xd8"),),
    )

    upload = respx.calls[0].request
    assert upload.url.path == "/files"
    assert b"image/jpeg" in upload.read()
    chat_body = respx.calls[1].request.read().decode()
    assert "file-api-abc" in chat_body
    # El binario crudo no viaja inline en la petición de chat.
    assert "file_data" not in chat_body


@respx.mock
async def test_error_de_files_api_se_propaga_como_ai_error():
    respx.post(FILES_URL).mock(return_value=httpx.Response(
        400, json={"error": {"message": "formato no soportado"}},
    ))
    client = make_client()

    with pytest.raises(AIError) as exc:
        await client.complete(
            [ChatMessage(MessageRole.USER, "describe")],
            attachments=(binary_payload("foto.jpg", b"\xff\xd8"),),
        )

    assert exc.value.user_message == "formato no soportado"


@respx.mock
async def test_varios_adjuntos_mantienen_su_orden():
    respx.post(URL).mock(return_value=completion())
    client = make_client()

    await client.complete(
        [ChatMessage(MessageRole.USER, "resume")],
        attachments=(
            text_payload("a.txt", "primero"),
            text_payload("b.txt", "segundo"),
        ),
    )

    body = respx.calls[0].request.read().decode()
    assert body.index("primero") < body.index("segundo")


@respx.mock
async def test_sin_adjuntos_el_payload_no_cambia():
    respx.post(URL).mock(return_value=completion())
    client = make_client()

    await client.complete([ChatMessage(MessageRole.USER, "hola")])

    body = respx.calls[0].request.read().decode()
    assert "file_id" not in body
    assert "file-api" not in body
