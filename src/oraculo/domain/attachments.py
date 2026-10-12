"""Modelos de dominio del ciclo de vida de archivos adjuntos.

Inmutables y sin dependencias de frameworks. Un adjunto siempre pertenece
a una sesión identificada por `SessionKey`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from oraculo.domain.errors import AttachmentError

# Tipos de origen que Telegram entrega como archivo adjunto.
INCOMING_KINDS = frozenset(
    {"document", "photo", "audio", "video", "voice", "animation"}
)


class AttachmentMode(Enum):
    """Cómo se envía el adjunto al proveedor de IA."""

    BINARY = "binary"
    TEXT = "text"


@dataclass(frozen=True)
class IncomingAttachment:
    """Archivo adjunto normalizado desde el adaptador de Telegram."""

    file_id: str
    file_name: str | None
    mime_type: str | None
    file_size: int | None
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.file_id, str) or not self.file_id:
            raise AttachmentError("invalid_attachment")
        if self.file_name is not None and not isinstance(self.file_name, str):
            raise AttachmentError("invalid_attachment")
        if self.mime_type is not None and not isinstance(self.mime_type, str):
            raise AttachmentError("invalid_attachment")
        if self.file_size is not None:
            if not isinstance(self.file_size, int) or self.file_size < 0:
                raise AttachmentError("invalid_attachment")
        if self.kind not in INCOMING_KINDS:
            raise AttachmentError("invalid_attachment")


@dataclass(frozen=True)
class StoredAttachment:
    """Adjunto ya descargado, clasificado y en la carpeta temporal."""

    name: str
    mime_type: str
    mode: AttachmentMode
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AttachmentError("invalid_attachment")
        if not isinstance(self.mime_type, str):
            raise AttachmentError("invalid_attachment")
        if not isinstance(self.mode, AttachmentMode):
            raise AttachmentError("invalid_attachment")
        if not isinstance(self.size, int) or self.size < 0:
            raise AttachmentError("invalid_attachment")
        mime_type = self.mime_type.split(";", 1)[0].strip().lower()
        if self.mode is AttachmentMode.BINARY and (
            not mime_type or mime_type == "application/octet-stream"
        ):
            raise AttachmentError("invalid_attachment")


@dataclass(frozen=True)
class AttachmentPayload:
    """Contenido listo para enviar al proveedor de IA."""

    name: str
    mime_type: str
    mode: AttachmentMode
    data: bytes | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise AttachmentError("invalid_payload")
        if not isinstance(self.mime_type, str):
            raise AttachmentError("invalid_payload")
        if self.mode is AttachmentMode.BINARY:
            mime_type = self.mime_type.split(";", 1)[0].strip().lower()
            if (
                not isinstance(self.data, bytes)
                or self.text is not None
                or not mime_type
                or mime_type == "application/octet-stream"
            ):
                raise AttachmentError("invalid_payload")
        elif self.mode is AttachmentMode.TEXT:
            if not isinstance(self.text, str) or self.data is not None:
                raise AttachmentError("invalid_payload")
        else:
            raise AttachmentError("invalid_payload")
