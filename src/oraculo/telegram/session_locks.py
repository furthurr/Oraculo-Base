"""Locks de entrada por conversación para serializar handlers del mismo chat."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from oraculo.domain.models import ChatKind, SessionKey


def session_key_for_chat(
    kind: ChatKind, chat_id: int, user_id: int
) -> SessionKey | None:
    """Mapea chats privados/grupo a las mismas claves de memoria conversacional."""
    if kind is ChatKind.PRIVATE:
        return SessionKey.private(user_id)
    if kind is ChatKind.GROUP:
        return SessionKey.group(chat_id)
    return None


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    references: int = 0


class SessionUpdateLockPool:
    """Comparte un lock por clave mientras existan titulares o tareas en espera."""

    def __init__(self) -> None:
        self._entries: dict[str, _LockEntry] = {}

    @asynccontextmanager
    async def hold(self, key: SessionKey) -> AsyncIterator[None]:
        """Serializa una operación por sesión y libera su entrada al quedar inactiva."""
        entry = self._entries.get(key.value)
        if entry is None:
            entry = _LockEntry(lock=asyncio.Lock())
            self._entries[key.value] = entry
        entry.references += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.references -= 1
            if entry.references == 0 and self._entries.get(key.value) is entry:
                del self._entries[key.value]
