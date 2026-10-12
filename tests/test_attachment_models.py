"""Tests de los modelos de dominio de adjuntos (Wave 1, tarea 1.1)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from oraculo.domain.attachments import (
    AttachmentMode,
    AttachmentPayload,
    IncomingAttachment,
    StoredAttachment,
)
from oraculo.domain.errors import AttachmentError


def test_attachment_mode_expone_binary_y_text():
    assert AttachmentMode.BINARY.value == "binary"
    assert AttachmentMode.TEXT.value == "text"


def test_incoming_attachment_es_inmutable():
    attachment = IncomingAttachment(
        file_id="file-1", file_name="a.txt", mime_type="text/plain",
        file_size=10, kind="document",
    )

    with pytest.raises(FrozenInstanceError):
        attachment.file_id = "otro"  # type: ignore[misc]


def test_incoming_attachment_rechaza_file_id_vacio():
    with pytest.raises(AttachmentError) as exc:
        IncomingAttachment(
            file_id="", file_name="a.txt", mime_type="text/plain",
            file_size=10, kind="document",
        )
    assert exc.value.reason == "invalid_attachment"


def test_incoming_attachment_rechaza_tamano_negativo():
    with pytest.raises(AttachmentError) as exc:
        IncomingAttachment(
            file_id="file-1", file_name="a.txt", mime_type="text/plain",
            file_size=-1, kind="document",
        )
    assert exc.value.reason == "invalid_attachment"


def test_incoming_attachment_rechaza_kind_desconocido():
    with pytest.raises(AttachmentError) as exc:
        IncomingAttachment(
            file_id="file-1", file_name="a.txt", mime_type="text/plain",
            file_size=10, kind="sticker",
        )
    assert exc.value.reason == "invalid_attachment"


def test_incoming_attachment_acepta_tamano_desconocido():
    attachment = IncomingAttachment(
        file_id="file-1", file_name=None, mime_type=None,
        file_size=None, kind="photo",
    )
    assert attachment.file_size is None


def test_stored_attachment_es_inmutable_y_valida():
    stored = StoredAttachment(
        name="a.txt", mime_type="text/plain", mode=AttachmentMode.TEXT, size=5,
    )
    with pytest.raises(FrozenInstanceError):
        stored.size = 6  # type: ignore[misc]
    with pytest.raises(AttachmentError):
        StoredAttachment(
            name="", mime_type="text/plain", mode=AttachmentMode.TEXT, size=5,
        )
    with pytest.raises(AttachmentError):
        StoredAttachment(
            name="a.txt", mime_type="text/plain", mode=AttachmentMode.TEXT, size=-1,
        )
    with pytest.raises(AttachmentError):
        StoredAttachment(
            name="foto.jpg", mime_type="application/octet-stream",
            mode=AttachmentMode.BINARY, size=5,
        )


def test_payload_binary_exige_datos_y_no_texto():
    payload = AttachmentPayload(
        name="foto.jpg", mime_type="image/jpeg",
        mode=AttachmentMode.BINARY, data=b"\xff\xd8",
    )
    assert payload.data == b"\xff\xd8"
    assert payload.text is None

    with pytest.raises(AttachmentError) as exc:
        AttachmentPayload(
            name="foto.jpg", mime_type="image/jpeg",
            mode=AttachmentMode.BINARY, text="hola",
        )
    assert exc.value.reason == "invalid_payload"

    with pytest.raises(AttachmentError):
        AttachmentPayload(
            name="foto.jpg", mime_type="image/jpeg",
            mode=AttachmentMode.BINARY,
        )


def test_payload_binary_rechaza_mime_generico():
    with pytest.raises(AttachmentError, match="invalid_payload"):
        AttachmentPayload(
            name="foto.jpg", mime_type="application/octet-stream",
            mode=AttachmentMode.BINARY, data=b"bytes",
        )


def test_payload_text_exige_texto_y_no_datos():
    payload = AttachmentPayload(
        name="a.txt", mime_type="text/plain",
        mode=AttachmentMode.TEXT, text="contenido",
    )
    assert payload.text == "contenido"
    assert payload.data is None

    with pytest.raises(AttachmentError) as exc:
        AttachmentPayload(
            name="a.txt", mime_type="text/plain",
            mode=AttachmentMode.TEXT, data=b"x",
        )
    assert exc.value.reason == "invalid_payload"

    with pytest.raises(AttachmentError):
        AttachmentPayload(
            name="a.txt", mime_type="text/plain", mode=AttachmentMode.TEXT,
        )


def test_attachment_error_expone_reason_sin_secretos():
    error = AttachmentError("too_large")
    assert error.reason == "too_large"
    assert "too_large" in str(error)
