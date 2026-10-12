"""Extracción acotada de mensajes de error de proveedores externos."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

import httpx

MAX_PROVIDER_ERROR_BODY_BYTES = 8 * 1024
MAX_PROVIDER_ERROR_MESSAGE_CHARS = 1_500
_READ_CHUNK_BYTES = 1_024

_TRACEBACK = re.compile(
    r"Traceback \(most recent call last\):|^\s*File \"[^\"]+\", line \d+",
    re.IGNORECASE | re.MULTILINE,
)
_BEARER_CREDENTIAL = re.compile(r"\b(Bearer|Basic)\s+[^\s,;]+", re.IGNORECASE)
_NAMED_CREDENTIAL = re.compile(
    r"\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|password|secret)"
    r"\s*[:=]\s*[^\s,;]+",
    re.IGNORECASE,
)
_TOKEN = re.compile(r"\b(?:sk|rk|pk|tp)-[A-Za-z0-9_-]{8,}\b")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")


async def read_bounded_error_body(response: httpx.Response) -> bytes | None:
    """Lee un cuerpo de error solo si cabe en el límite descomprimido."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes(chunk_size=_READ_CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_PROVIDER_ERROR_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def extract_provider_error_message(
    payload: object,
    *,
    secrets: Sequence[str] = (),
) -> str | None:
    """Extrae campos de error conocidos y devuelve solo texto filtrado.

    Acepta bytes JSON de una respuesta HTTP o un mapping ya decodificado de un
    evento SSE. No serializa mappings ni comparte campos desconocidos.
    """
    value = payload
    if isinstance(payload, bytes):
        if len(payload) > MAX_PROVIDER_ERROR_BODY_BYTES:
            return None
        try:
            value = json.loads(payload)
        except (ValueError, UnicodeDecodeError):
            return None

    candidate = _candidate_message(value)
    if candidate is None:
        return None
    try:
        candidate_size = len(candidate.encode("utf-8"))
    except UnicodeEncodeError:
        return None
    if candidate_size > MAX_PROVIDER_ERROR_BODY_BYTES:
        return None
    return _sanitize_message(candidate, secrets)


def _candidate_message(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None

    containers: list[Mapping[str, object]] = [value]
    response = value.get("response")
    if isinstance(response, Mapping):
        containers.append(response)

    for container in containers:
        error = container.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, Mapping):
            message = error.get("message")
            if isinstance(message, str):
                return message
        message = container.get("message")
        if isinstance(message, str):
            return message
    return None


def _sanitize_message(candidate: str, secrets: Sequence[str]) -> str | None:
    if _TRACEBACK.search(candidate):
        return None

    message = candidate.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    message = _CONTROL.sub(" ", message).strip()
    for secret in sorted((value for value in secrets if value), key=len, reverse=True):
        message = message.replace(secret, "[redactado]")
    message = _BEARER_CREDENTIAL.sub(r"\1 [redactado]", message)
    message = _NAMED_CREDENTIAL.sub(r"\1=[redactado]", message)
    message = _TOKEN.sub("[credencial redactada]", message)
    if not message or not message.replace("[redactado]", "").replace(
        "[credencial redactada]", ""
    ).strip():
        return None
    if len(message) > MAX_PROVIDER_ERROR_MESSAGE_CHARS:
        message = message[:MAX_PROVIDER_ERROR_MESSAGE_CHARS - 1].rstrip() + "…"
    return message
