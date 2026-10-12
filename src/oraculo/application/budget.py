"""Reparto de adjuntos en tandas según el presupuesto de petición del proveedor.

Ninguna tanda supera `max_bytes`, que representa el límite documentado del
proveedor por petición (p. ej. 40 MB para GPT, bajo el tope de 50 MB). Un
archivo que excede el presupuesto va solo en su propia tanda: el proveedor lo
rechazará si procede y ese error se muestra al usuario.
"""

from __future__ import annotations

from collections.abc import Sequence

from oraculo.domain.attachments import AttachmentMode, AttachmentPayload

# Presupuesto por defecto y documentado para GPT; también aplica a los demás.
DEFAULT_REQUEST_BUDGET_BYTES = 40_000_000


def payload_size(payload: AttachmentPayload) -> int:
    """Bytes que ocupa el adjunto dentro de la petición."""
    if payload.mode is AttachmentMode.BINARY:
        # Base64 expande 4/3; el texto extraído viaja ya truncado.
        return int(len(payload.data or b"") * 4 / 3)
    return len((payload.text or "").encode("utf-8"))


def split_by_budget(
    payloads: Sequence[AttachmentPayload], *, max_bytes: int
) -> list[tuple[AttachmentPayload, ...]]:
    """Reparte los adjuntos en tandas que respetan el presupuesto.

    Conserva el orden global de llegada. Con adjuntos vacíos devuelve `[]`.
    """
    if max_bytes < 1:
        raise ValueError("max_bytes debe ser positivo")
    tandas: list[tuple[AttachmentPayload, ...]] = []
    current: list[AttachmentPayload] = []
    used = 0
    for payload in payloads:
        size = payload_size(payload)
        if current and used + size > max_bytes:
            tandas.append(tuple(current))
            current = []
            used = 0
        current.append(payload)
        used += size
    if current:
        tandas.append(tuple(current))
    return tandas


def merge_replies(replies: Sequence[str]) -> str:
    """Concatena las respuestas de cada tanda, con aviso si hubo varias."""
    non_empty = [reply for reply in replies if reply]
    if not non_empty:
        return ""
    if len(non_empty) == 1:
        return non_empty[0]
    body = "\n\n".join(
        f"[Archivos — parte {index} de {len(non_empty)}]\n{reply}"
        for index, reply in enumerate(non_empty, start=1)
    )
    return (
        "Los archivos no cabían en una sola consulta, así que los he procesado "
        f"por tandas:\n\n{body}"
    )
