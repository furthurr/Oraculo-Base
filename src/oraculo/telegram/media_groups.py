"""Agrupación de álbumes de Telegram en una única interacción.

Telegram entrega cada archivo de un álbum (media group) como un update
independiente con el mismo `media_group_id`. Este buffer los acumula durante
una ventana breve y los entrega juntos, en orden de llegada, cuando la
ventana se cierra.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from oraculo.domain.attachments import IncomingAttachment

DEFAULT_WINDOW_S = 2.0


@dataclass
class _PendingGroup:
    attachments: list[IncomingAttachment] = field(default_factory=list)
    deadline: float = 0.0


class MediaGroupBuffer:
    """Acumula updates de un mismo álbum y entrega la interacción completa."""

    def __init__(self, *, window_s: float = DEFAULT_WINDOW_S) -> None:
        if window_s <= 0:
            raise ValueError("window_s debe ser positivo")
        self._window_s = window_s
        self._groups: dict[tuple[int, str], _PendingGroup] = {}

    def add(
        self,
        *,
        chat_id: int,
        media_group_id: str | None,
        attachment: IncomingAttachment,
    ) -> tuple[IncomingAttachment, ...] | None:
        """Registra un adjunto; devuelve la interacción si no pertenece a álbum."""
        if media_group_id is None:
            # Sin álbum la interacción es el propio mensaje: se entrega ya.
            return (attachment,)
        key = (chat_id, media_group_id)
        group = self._groups.get(key)
        if group is None:
            group = _PendingGroup(deadline=time.monotonic() + self._window_s)
            self._groups[key] = group
        group.attachments.append(attachment)
        return None

    def flush_ready(self) -> list[tuple[int, str | None, tuple[IncomingAttachment, ...]]]:
        """Devuelve las interacciones cuya ventana ya se cerró."""
        ready: list[tuple[int, str | None, tuple[IncomingAttachment, ...]]] = []
        now = time.monotonic()
        for key, group in list(self._groups.items()):
            if now >= group.deadline:
                chat_id, media_group_id = key
                ready.append((chat_id, media_group_id, tuple(group.attachments)))
                del self._groups[key]
        return ready

    def flush_group(
        self, *, chat_id: int, media_group_id: str
    ) -> tuple[int, str, tuple[IncomingAttachment, ...]] | None:
        """Vacía solo el álbum indicado si venció su ventana de agrupación."""
        key = (chat_id, media_group_id)
        group = self._groups.get(key)
        if group is None or time.monotonic() < group.deadline:
            return None
        del self._groups[key]
        return chat_id, media_group_id, tuple(group.attachments)

    def contains_group(self, *, chat_id: int, media_group_id: str) -> bool:
        """Indica si el álbum indicado sigue esperando a que venza su ventana."""
        return (chat_id, media_group_id) in self._groups
