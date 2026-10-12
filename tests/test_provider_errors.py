from __future__ import annotations

import httpx

from oraculo.infrastructure.provider_errors import (
    MAX_PROVIDER_ERROR_BODY_BYTES,
    MAX_PROVIDER_ERROR_MESSAGE_CHARS,
    extract_provider_error_message,
    read_bounded_error_body,
)


def test_extrae_solo_el_mensaje_de_error_openai_compatible() -> None:
    payload = b'{"error":{"message":"cuota agotada","type":"rate_limit"}}'

    assert extract_provider_error_message(payload) == "cuota agotada"


def test_extrae_error_textual_y_mensaje_anidado_de_codex() -> None:
    assert extract_provider_error_message({"error": "rechazo del proveedor"}) == (
        "rechazo del proveedor"
    )
    assert extract_provider_error_message(
        {"type": "response.failed", "response": {"error": {"message": "sin cuota"}}}
    ) == "sin cuota"


def test_ignora_cuerpos_invalidos_o_sin_mensaje_reconocido() -> None:
    assert extract_provider_error_message(b"<html>error</html>") is None
    assert extract_provider_error_message(b'{"error":{"code":"bad_request"}}') is None
    assert extract_provider_error_message({"error": 42}) is None
    assert extract_provider_error_message(b'{"error":{"message":"\\ud800"}}') is None


def test_filtra_credenciales_conocidas_y_patrones_comunes() -> None:
    key = "sk-1234567890abcdef"
    bearer = "access-SENTINEL"
    message = extract_provider_error_message(
        {"error": {"message": f"API key {key} rechazada; Bearer {bearer}"}},
        secrets=(bearer,),
    )

    assert message is not None
    assert key not in message
    assert bearer not in message
    assert "API key" in message


def test_descarta_trazas_y_limita_el_texto_visible() -> None:
    traceback = "Traceback (most recent call last):\\n  File '/srv/app.py', line 10"
    assert extract_provider_error_message({"message": traceback}) is None

    message = extract_provider_error_message({"message": "x" * (MAX_PROVIDER_ERROR_MESSAGE_CHARS + 50)})
    assert message is not None
    assert len(message) == MAX_PROVIDER_ERROR_MESSAGE_CHARS
    assert message.endswith("…")


def test_no_inspecciona_un_cuerpo_mayor_al_limite() -> None:
    body = b'{"error":{"message":"' + b"x" * MAX_PROVIDER_ERROR_BODY_BYTES + b'"}}'

    assert extract_provider_error_message(body) is None


async def test_lee_respuestas_de_error_solo_hasta_el_limite() -> None:
    small = httpx.Response(400, content=b'{"error":"no autorizado"}')
    large = httpx.Response(500, content=b"x" * (MAX_PROVIDER_ERROR_BODY_BYTES + 1))

    assert await read_bounded_error_body(small) == b'{"error":"no autorizado"}'
    assert await read_bounded_error_body(large) is None
