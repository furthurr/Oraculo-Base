"""Gestión de sesiones conversacionales: ventana, expiración y registro."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from oraculo.domain.models import (
    ChatKind,
    ChatMessage,
    MessageRole,
    SessionKey,
    SessionState,
)
from oraculo.domain.errors import AIError
from oraculo.domain.ports import AIClient, SessionRepository

logger = logging.getLogger(__name__)

# Tope de locks en memoria (mismo orden de magnitud que las sesiones).
MAX_LOCKS = 1000
MAX_SUMMARY_CHARS = 6000


def with_author(kind: ChatKind, message: ChatMessage) -> ChatMessage:
    """Prefija el nombre del autor en los mensajes de usuario de un grupo."""
    if kind is not ChatKind.GROUP or message.role is not MessageRole.USER:
        return message
    if not message.author_name:
        return message
    return replace(message, content=f"{message.author_name}: {message.content}")


class SessionManager:
    """Ciclo de vida de las sesiones sobre un `SessionRepository`."""

    def __init__(
        self,
        *,
        repository: SessionRepository,
        max_turns: int,
        idle_minutes: int,
        clock: Callable[[], datetime],
    ) -> None:
        self._repository = repository
        self._max_turns = max_turns
        self._idle_minutes = idle_minutes
        self._clock = clock
        self._locks: dict[str, asyncio.Lock] = {}

    @property
    def enabled(self) -> bool:
        """La memoria está activa salvo que `SESSION_MAX_TURNS` sea 0."""
        return self._max_turns > 0

    def lock_for(self, key: SessionKey) -> asyncio.Lock:
        """Devuelve el lock que serializa los mensajes de una misma sesión."""
        lock = self._locks.get(key.value)
        if lock is None:
            self._prune_locks()
            lock = asyncio.Lock()
            self._locks[key.value] = lock
        return lock

    async def load_window(self, key: SessionKey) -> tuple[ChatMessage, ...]:
        """Últimos turnos de la sesión, listos para enviar (con atribución)."""
        if not self.enabled:
            return ()
        state = await self._repository.load(key)
        if state is None:
            return ()
        if self._expired(state, self._clock()):
            await self._repository.discard(key)
            logger.info(
                "Sesión expirada key=%s idle_minutes=%s",
                key.value,
                self._idle_minutes,
            )
            return ()
        return self._window(key, state)

    async def compact_if_due(self, key: SessionKey, ai: AIClient) -> bool:
        """Resume un tramo completo; el caller mantiene el lock por sesión."""
        if not self.enabled:
            return False
        state = await self._repository.load(key)
        if state is None or len(state.messages) < 2 * self._max_turns:
            return False
        try:
            summary = await ai.summarize(
                state.summary,
                self._window_messages(key, state),
                max_chars=MAX_SUMMARY_CHARS,
            )
            if not summary.strip() or len(summary) > MAX_SUMMARY_CHARS:
                raise AIError("resumen inválido")
        except AIError:
            logger.warning("No se pudo resumir sesión key=%s", key.value)
            return False
        await self._repository.compact(key, summary)
        return True

    async def compaction_due(self, key: SessionKey) -> bool:
        if not self.enabled:
            return False
        state = await self._repository.load(key)
        return state is not None and len(state.messages) >= 2 * self._max_turns

    async def record(
        self,
        key: SessionKey,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
    ) -> None:
        """Registra un turno completo (solo tras una respuesta exitosa)."""
        if not self.enabled:
            return
        await self._repository.append(
            key, user_message, assistant_message, now=self._clock()
        )

    async def reset(self, key: SessionKey) -> None:
        """Descarta la sesión de la clave indicada."""
        if not self.enabled:
            return
        await self._repository.discard(key)
        logger.info("Sesión reiniciada key=%s", key.value)

    def _expired(self, state: SessionState, now: datetime) -> bool:
        if self._idle_minutes <= 0:
            return False
        elapsed_minutes = (now - state.updated_at).total_seconds() / 60
        return elapsed_minutes > self._idle_minutes

    def _window(self, key: SessionKey, state: SessionState) -> tuple[ChatMessage, ...]:
        messages = self._window_messages(key, state)
        if state.summary:
            return (
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=f"[Resumen acumulado de la conversación]\n{state.summary}",
                ),
                *messages,
            )
        return messages

    def _window_messages(self, key: SessionKey, state: SessionState) -> tuple[ChatMessage, ...]:
        messages = state.messages[-(self._max_turns * 2) :]
        while messages and messages[0].role is not MessageRole.USER:
            messages = messages[1:]
        return tuple(with_author(key.kind, message) for message in messages)

    def _prune_locks(self) -> None:
        if len(self._locks) < MAX_LOCKS:
            return
        for value in [
            value for value, lock in self._locks.items() if not lock.locked()
        ]:
            del self._locks[value]
