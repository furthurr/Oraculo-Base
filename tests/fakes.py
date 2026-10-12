"""Dobles de prueba compartidos."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Optional

from oraculo.domain.errors import AIError
from oraculo.domain.models import (
    ChatKind,
    ChatMessage,
    IncomingMessage,
    SessionKey,
    SessionState,
    SkillTurnContext,
)
from oraculo.domain.pi import PiModeError, PiProject
from oraculo.domain.ports import EMPTY_SKILL_CONTEXT


class FakeAIClient:
    """Cliente de IA falso que registra las conversaciones recibidas."""

    def __init__(
        self, reply: str = "respuesta IA", error: Optional[AIError] = None
    ) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[ChatMessage, ...]] = []
        self.skill_contexts: list[SkillTurnContext] = []
        self.attachments_calls: list[tuple[object, ...]] = []
        self.summaries: list[tuple[str | None, tuple[ChatMessage, ...], int]] = []
        self.summary_reply = "resumen generado"
        self.summary_error: AIError | None = None

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        skills: SkillTurnContext = EMPTY_SKILL_CONTEXT,
        workspace_tools: object | None = None,
        attachments: Sequence[object] = (),
    ) -> str:
        self.calls.append(tuple(messages))
        self.skill_contexts.append(skills)
        self.attachments_calls.append(tuple(attachments))
        # Cede el control: expone carreras de concurrencia en los tests.
        await asyncio.sleep(0)
        if self.error is not None:
            raise self.error
        return self.reply

    async def summarize(
        self, previous_summary: str | None, messages: Sequence[ChatMessage], *, max_chars: int
    ) -> str:
        self.summaries.append((previous_summary, tuple(messages), max_chars))
        await asyncio.sleep(0)
        if self.summary_error:
            raise self.summary_error
        return self.summary_reply


class FakeSessionRepository:
    """Repositorio de sesiones en memoria para pruebas, con registro de llamadas."""

    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self.appended: list[tuple[str, ChatMessage, ChatMessage]] = []
        self.discarded: list[str] = []

    async def load(self, key: SessionKey) -> SessionState | None:
        return self.sessions.get(key.value)

    async def append(
        self,
        key: SessionKey,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        *,
        now: datetime,
    ) -> None:
        self.appended.append((key.value, user_message, assistant_message))
        current = self.sessions.get(key.value)
        messages = (current.messages if current is not None else ()) + (
            user_message,
            assistant_message,
        )
        self.sessions[key.value] = SessionState(
            messages=messages, updated_at=now,
            summary=current.summary if current else None,
        )

    async def compact(self, key: SessionKey, summary: str) -> None:
        current = self.sessions[key.value]
        self.sessions[key.value] = SessionState((), current.updated_at, summary)

    async def discard(self, key: SessionKey) -> None:
        self.discarded.append(key.value)
        self.sessions.pop(key.value, None)


class FakeClock:
    """Reloj controlable para pruebas de expiración."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, delta: timedelta) -> None:
        self.now += delta


class FakeAccessPolicy:
    def __init__(
        self,
        users: frozenset[int] = frozenset(),
        chats: frozenset[int] = frozenset(),
    ) -> None:
        self.users = users
        self.chats = chats

    def is_user_allowed(self, user_id: int) -> bool:
        return user_id in self.users

    def is_chat_allowed(self, chat_id: int) -> bool:
        return chat_id in self.chats


class FakePhraseProvider:
    def __init__(
        self,
        rejection: str = "No hablo con extraños",
        only_text: str = "solo texto",
        empty_mention: str = "¿qué quieres saber?",
        new_session: str = "sesión nueva",
        skill_limit: str = "límite de skills",
    ) -> None:
        self.rejection = rejection
        self.only_text = only_text
        self.empty_mention = empty_mention
        self.new_session = new_session
        self.skill_limit = skill_limit
        self.rejection_calls = 0
        self.only_text_calls = 0
        self.empty_mention_calls = 0
        self.new_session_calls = 0
        self.skill_limit_calls = 0

    def rejection_phrase(self) -> str:
        self.rejection_calls += 1
        return self.rejection

    def only_text_phrase(self) -> str:
        self.only_text_calls += 1
        return self.only_text

    def empty_mention_phrase(self) -> str:
        self.empty_mention_calls += 1
        return self.empty_mention

    def new_session_phrase(self) -> str:
        self.new_session_calls += 1
        return self.new_session

    def skill_limit_phrase(self) -> str:
        self.skill_limit_calls += 1
        return self.skill_limit


def private_message(
    text: Optional[str] = "hola",
    user_id: int = 1,
    mentions_bot: bool = False,
    replies_to_bot: bool = False,
    quoted_text: Optional[str] = None,
    user_display_name: Optional[str] = None,
) -> IncomingMessage:
    return IncomingMessage(
        chat_id=100,
        user_id=user_id,
        chat_type=ChatKind.PRIVATE,
        text=text,
        mentions_bot=mentions_bot,
        replies_to_bot=replies_to_bot,
        quoted_text=quoted_text,
        user_display_name=user_display_name,
    )


def group_message(
    text: Optional[str] = "hola",
    chat_id: int = -100,
    user_id: int = 1,
    mentions_bot: bool = True,
    replies_to_bot: bool = False,
    quoted_text: Optional[str] = None,
    user_display_name: Optional[str] = None,
) -> IncomingMessage:
    return IncomingMessage(
        chat_id=chat_id,
        user_id=user_id,
        chat_type=ChatKind.GROUP,
        text=text,
        mentions_bot=mentions_bot,
        replies_to_bot=replies_to_bot,
        quoted_text=quoted_text,
        user_display_name=user_display_name,
    )


class FakeProjectStore:
    """Almacén de proyectos Pi en memoria; implementa `PiProjectStorePort`."""

    ROOT = "/fake/pi"

    def __init__(
        self,
        names: Sequence[str] = ("demo",),
        *,
        list_error: Exception | None = None,
        create_error: Exception | None = None,
    ) -> None:
        self.names = sorted(names)
        self.list_error = list_error
        self.create_error = create_error
        self.created: list[str] = []

    def _project(self, name: str) -> PiProject:
        return PiProject(name=name, path=f"{self.ROOT}/{name}")

    def list_projects(self) -> list[PiProject]:
        if self.list_error is not None:
            raise self.list_error
        return [self._project(n) for n in self.names]

    def resolve(self, name: str) -> PiProject:
        if name not in self.names:
            raise PiModeError("project_invalid", user_message=f"El proyecto «{name}» no está disponible.")
        return self._project(name)

    def create(self, name: str) -> tuple[PiProject, bool]:
        if self.create_error is not None:
            raise self.create_error
        if not name or "/" in name or name.startswith("."):
            raise PiModeError("project_invalid", user_message="Nombre de proyecto no válido.")
        if name in self.names:
            return self._project(name), True
        self.names = sorted([*self.names, name])
        self.created.append(name)
        return self._project(name), False


async def enter_pi(service: object, user_id: int, choice: str = "1") -> list[object]:
    """Entra en modo Pi: `/agent` y selección del proyecto por número."""
    events = [e async for e in service.handle(user_id, "/agent")]  # type: ignore[attr-defined]
    events += [e async for e in service.handle(user_id, choice)]  # type: ignore[attr-defined]
    return events
