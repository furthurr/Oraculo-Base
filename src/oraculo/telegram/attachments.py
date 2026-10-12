"""Normalización de adjuntos entrantes desde el adaptador de Telegram.

Extrae los archivos de un mensaje (documentos, fotos, audio, vídeo, voz y
animaciones) a `IncomingAttachment`, sin I/O de red: la descarga es
responsabilidad del caller.
"""

from __future__ import annotations

import logging

from telegram.error import TelegramError

from oraculo.application.attachments import MAX_FILE_BYTES
from oraculo.domain.attachments import IncomingAttachment

logger = logging.getLogger(__name__)

_KINDS = (
    ("document", "document"),
    ("photo", "photo"),
    ("audio", "audio"),
    ("video", "video"),
    ("voice", "voice"),
    ("animation", "animation"),
)


def _from_media(kind: str, media: object) -> IncomingAttachment | None:
    file_id = getattr(media, "file_id", None)
    if not isinstance(file_id, str) or not file_id:
        return None
    return IncomingAttachment(
        file_id=file_id,
        file_name=getattr(media, "file_name", None),
        mime_type=getattr(media, "mime_type", None),
        file_size=getattr(media, "file_size", None),
        kind=kind,
    )


def normalize_attachments(message: object) -> tuple[IncomingAttachment, ...]:
    """Devuelve los adjuntos del mensaje en el orden de los tipos soportados."""
    found: list[IncomingAttachment] = []
    for attribute, kind in _KINDS:
        value = getattr(message, attribute, None)
        if value is None:
            continue
        # `photo` llega como lista de tamaños: se conserva el mayor.
        candidates = value if isinstance(value, (list, tuple)) else (value,)
        if kind == "photo":
            candidates = sorted(
                candidates,
                key=lambda item: getattr(item, "file_size", 0) or 0,
            )
        for item in candidates:
            attachment = _from_media(kind, item)
            if attachment is not None:
                if kind == "photo":
                    found = [a for a in found if a.kind != "photo"]
                found.append(attachment)
    return tuple(found)


async def download_attachment(
    bot: object, attachment: IncomingAttachment, *, max_bytes: int = MAX_FILE_BYTES
) -> bytes | None:
    """Descarga el archivo desde la API de Bot, o `None` si no procede.

    El peso declarado se valida antes de llamar a `getFile`. El caller valida el
    tamaño real descargado para poder informar correctamente si excede el límite.
    Un fallo del Bot API se registra sin incluir identificadores del archivo.
    """
    if attachment.file_size is not None and attachment.file_size > max_bytes:
        return None
    try:
        file = await bot.get_file(attachment.file_id)
        data = await file.download_as_bytearray()
    except TelegramError:
        logger.warning(
            "No se pudo descargar el adjunto kind=%s",
            attachment.kind,
        )
        return None
    return bytes(data)
