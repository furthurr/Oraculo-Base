"""Adaptador `SessionRepository` en memoria (dict LRU acotado)."""

from __future__ import annotations

import logging
from collections import OrderedDict
from datetime import datetime

from oraculo.domain.models import ChatMessage, SessionKey, SessionState

logger = logging.getLogger(__name__)

# Topes de seguridad (constantes de diseño; ver design.md). No configurables.
MAX_STORED_TURNS = 100
MAX_SESSIONS = 1000


class InMemorySessionRepository:
    """Sesiones en RAM; se pierden al reiniciar el proceso.

    - Tope por sesión: descarta los turnos más antiguos al exceder el máximo.
    - Tope global: descarta la sesión de uso menos reciente (LRU).
    """

    def __init__(
        self,
        *,
        max_stored_turns: int = MAX_STORED_TURNS,
        max_sessions: int = MAX_SESSIONS,
    ) -> None:
        self._max_messages = max_stored_turns * 2
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, SessionState] = OrderedDict()

    async def load(self, key: SessionKey) -> SessionState | None:
        state = self._sessions.get(key.value)
        if state is not None:
            self._sessions.move_to_end(key.value)
        return state

    async def append(
        self,
        key: SessionKey,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        *,
        now: datetime,
    ) -> None:
        current = self._sessions.get(key.value)
        if current is None:
            logger.debug("Sesión creada key=%s", key.value)
        messages = (current.messages if current is not None else ()) + (
            user_message,
            assistant_message,
        )
        if len(messages) > self._max_messages:
            messages = messages[-self._max_messages :]
        self._sessions[key.value] = SessionState(
            messages=messages, updated_at=now,
            summary=current.summary if current is not None else None,
        )
        self._sessions.move_to_end(key.value)
        while len(self._sessions) > self._max_sessions:
            self._sessions.popitem(last=False)

    async def discard(self, key: SessionKey) -> None:
        self._sessions.pop(key.value, None)

    async def compact(self, key: SessionKey, summary: str) -> None:
        current = self._sessions[key.value]
        self._sessions[key.value] = SessionState(
            messages=(), updated_at=current.updated_at, summary=summary,
        )
        self._sessions.move_to_end(key.value)
