"""Puertos de salida del dominio (implementados por infraestructura)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from oraculo.domain.attachments import AttachmentMode, AttachmentPayload
from oraculo.domain.models import (
    AttachmentKey,
    ChatMessage,
    SessionKey,
    SessionState,
    SkillCatalogSnapshot,
    SkillTurnContext,
)
from oraculo.domain.workspace import DocumentRef, DocumentVersion, WorkspaceScope

EMPTY_SKILL_CONTEXT = SkillTurnContext.empty()


@runtime_checkable
class WorkspaceToolExecutor(Protocol):
    """Ejecutor de herramientas documentales autorizado para un único turno."""

    async def execute(self, name: str, arguments_json: str) -> dict[str, object]:
        """Valida y ejecuta una llamada, devolviendo un resultado serializable."""


@runtime_checkable
class DocumentStore(Protocol):
    """Persistencia de documentos confinada a un ámbito por la implementación."""

    async def list(self, scope: WorkspaceScope) -> tuple[DocumentRef, ...]: ...

    async def read(self, scope: WorkspaceScope, ref: DocumentRef) -> tuple[str, DocumentVersion]: ...

    async def create(self, scope: WorkspaceScope, ref: DocumentRef, content: str) -> None: ...

    async def replace_if_version(
        self, scope: WorkspaceScope, ref: DocumentRef, content: str, version: DocumentVersion,
    ) -> None: ...

    async def attachment(self, scope: WorkspaceScope, ref: DocumentRef) -> bytes: ...


@runtime_checkable
class AttachmentStore(Protocol):
    """Almacenamiento temporal de adjuntos por chat y usuario."""

    async def store(
        self,
        key: AttachmentKey,
        name: str,
        mime_type: str,
        mode: AttachmentMode,
        data: bytes,
    ) -> None:
        """Guarda los bytes de un adjunto bajo el nombre saneado dado."""

    async def load(self, key: AttachmentKey, name: str) -> bytes:
        """Devuelve los bytes del adjunto o lanza `AttachmentError`."""

    async def list(self, key: AttachmentKey) -> tuple[str, ...]:
        """Nombres de los adjuntos vigentes de la sesión."""

    async def delete(self, key: AttachmentKey, name: str) -> None:
        """Elimina un adjunto; sin error si no existe."""

    async def clear(self, key: AttachmentKey) -> None:
        """Elimina todos los adjuntos de la sesión."""

    async def clear_all(self) -> None:
        """Elimina todos los adjuntos de todas las sesiones (arranque)."""


@runtime_checkable
class AIClient(Protocol):
    """Cliente de un modelo de IA que responde a una conversación."""

    async def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        skills: SkillTurnContext = EMPTY_SKILL_CONTEXT,
        workspace_tools: WorkspaceToolExecutor | None = None,
        attachments: Sequence[AttachmentPayload] = (),
    ) -> str:
        """Devuelve la respuesta del modelo o lanza `AIError`."""

    async def summarize(
        self,
        previous_summary: str | None,
        messages: Sequence[ChatMessage],
        *,
        max_chars: int,
    ) -> str:
        """Resume una sesión sin skills o tools; lanza `AIError` ante fallo."""


@runtime_checkable
class AudioTranscriber(Protocol):
    """Transcribe bytes de audio completos a texto, o lanza un error tipado."""

    async def transcribe(self, data: bytes, *, mime_type: str) -> str:
        """Devuelve una transcripción no vacía del audio."""


@runtime_checkable
class SkillCatalog(Protocol):
    """Fuente de un snapshot inmutable de skills cargado al arrancar."""

    @property
    def snapshot(self) -> SkillCatalogSnapshot:
        """Devuelve siempre el mismo snapshot durante la ejecución."""


@runtime_checkable
class SessionRepository(Protocol):
    """Almacén de sesiones conversacionales.

    La implementación actual vive en memoria; una futura persistencia (SQLite)
    será otro adaptador de este mismo puerto.
    """

    async def load(self, key: SessionKey) -> SessionState | None:
        """Devuelve la sesión o `None` si no existe."""

    async def append(
        self,
        key: SessionKey,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        *,
        now: datetime,
    ) -> None:
        """Agrega un turno completo y actualiza la última actividad."""

    async def discard(self, key: SessionKey) -> None:
        """Elimina la sesión de la clave indicada si existe."""

    async def compact(self, key: SessionKey, summary: str) -> None:
        """Sustituye de forma atómica el tramo bruto por su resumen acumulado."""


@runtime_checkable
class AccessPolicy(Protocol):
    """Decide qué usuarios y chats pueden usar el bot."""

    def is_user_allowed(self, user_id: int) -> bool:
        """Indica si el usuario puede conversar en privado."""

    def is_chat_allowed(self, chat_id: int) -> bool:
        """Indica si el chat está en la lista de chats permitidos."""


@runtime_checkable
class PhraseProvider(Protocol):
    """Provee frases aleatorias para las respuestas automáticas del bot."""

    def rejection_phrase(self) -> str:
        """Devuelve una frase aleatoria para usuarios no permitidos."""

    def only_text_phrase(self) -> str:
        """Devuelve una frase aleatoria para mensajes que no son de texto."""

    def empty_mention_phrase(self) -> str:
        """Devuelve una frase aleatoria cuando solo mencionan al bot."""

    def new_session_phrase(self) -> str:
        """Devuelve una frase aleatoria de confirmación de sesión nueva."""

    def skill_limit_phrase(self) -> str:
        """Devuelve la frase local para límites de skills."""
