"""Tests de contrato de los puertos de adjuntos (Wave 1, tarea 1.2)."""

from __future__ import annotations

import inspect

import pytest

from oraculo.domain.attachments import (
    AttachmentMode,
    AttachmentPayload,
    IncomingAttachment,
)
from oraculo.domain.models import AttachmentKey, ChatKind, SessionKey
from oraculo.domain.ports import AIClient, AttachmentStore, EMPTY_SKILL_CONTEXT


class _MinimalStore:
    """Implementación mínima del puerto `AttachmentStore`."""

    async def store(self, key, name, mime_type, mode, data):  # type: ignore[no-untyped-def]
        return None

    async def load(self, key, name):  # type: ignore[no-untyped-def]
        return b""

    async def list(self, key):  # type: ignore[no-untyped-def]
        return ()

    async def delete(self, key, name):  # type: ignore[no-untyped-def]
        return None

    async def clear(self, key):  # type: ignore[no-untyped-def]
        return None

    async def clear_all(self):  # type: ignore[no-untyped-def]
        return None


class _MinimalAI:
    """Implementación mínima del puerto `AIClient` con attachments."""

    async def complete(self, messages, *, skills=EMPTY_SKILL_CONTEXT,
                       workspace_tools=None, attachments=()):  # type: ignore[no-untyped-def]
        return "ok"

    async def summarize(self, previous_summary, messages, *, max_chars):  # type: ignore[no-untyped-def]
        return "ok"


def test_attachment_store_es_runtime_checkable():
    assert isinstance(_MinimalStore(), AttachmentStore)
    assert not isinstance(object(), AttachmentStore)


def test_ai_client_acepta_attachments_en_complete():
    client = _MinimalAI()
    assert isinstance(client, AIClient)
    signature = inspect.signature(_MinimalAI.complete)
    assert "attachments" in signature.parameters
    assert signature.parameters["attachments"].default == ()


def test_incoming_attachment_circula_hasta_el_payload():
    incoming = IncomingAttachment(
        file_id="file-1", file_name="a.txt", mime_type="text/plain",
        file_size=3, kind="document",
    )
    payload = AttachmentPayload(
        name=incoming.file_name or "sin-nombre",
        mime_type=incoming.mime_type or "application/octet-stream",
        mode=AttachmentMode.TEXT, text="abc",
    )
    assert payload.name == "a.txt"
    assert SessionKey.private(1).value == "private:1"
    attachment_key = AttachmentKey(kind=ChatKind.GROUP, chat_id=-10, user_id=1)
    assert attachment_key.value == "group:-10:1"
