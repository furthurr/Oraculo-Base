"""Modelos de dominio (estructuras inmutables en memoria)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from oraculo.domain.errors import SkillLimitError, SkillLoadError


class ChatKind(Enum):
    """Tipo de chat de Telegram que nos interesa distinguir."""

    PRIVATE = "private"
    GROUP = "group"
    OTHER = "other"


class MessageRole(Enum):
    """Rol de un mensaje dentro de una conversación enviada a la IA."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    """Mensaje de una conversación.

    `author_name` identifica a quien lo escribió; solo se usa para atribuir los
    mensajes de usuario en las sesiones de grupo.
    """

    role: MessageRole
    content: str
    author_name: str | None = None


@dataclass(frozen=True)
class Skill:
    """Instrucciones declarativas de una skill, sin información del filesystem."""

    name: str
    description: str
    instructions: str
    content_hash: str
    license: str | None = None
    compatibility: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class SkillCatalogSnapshot:
    """Vista ordenada e inmutable de las skills cargadas al arrancar."""

    skills: tuple[Skill, ...] = ()

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.skills, key=lambda skill: skill.name))
        names = tuple(skill.name for skill in ordered)
        if len(names) != len(set(names)):
            raise SkillLoadError("el snapshot contiene identificadores duplicados")
        object.__setattr__(self, "skills", ordered)

    def find(self, name: str) -> Skill | None:
        """Busca una skill por su identificador sin exponer índices mutables."""
        return next((skill for skill in self.skills if skill.name == name), None)


@dataclass(frozen=True)
class SkillTurnContext:
    """Contexto efímero y acotado de skills para una consulta a la IA."""

    catalog_text: str = ""
    explicit_skills: tuple[Skill, ...] = ()
    loadable_skills: tuple[Skill, ...] = ()
    autoload_enabled: bool = False
    max_rounds: int = 1
    max_context_chars: int = 1

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise SkillLimitError("max_rounds debe ser mayor que cero")
        if self.max_context_chars < 1:
            raise SkillLimitError("max_context_chars debe ser mayor que cero")

    @classmethod
    def empty(cls) -> SkillTurnContext:
        """Contexto neutral que conserva el comportamiento previo del cliente."""
        return cls()


@dataclass(frozen=True)
class SkillPreparation:
    """Separa el prompt persistible del contexto efímero enviado a la IA."""

    prompt: str
    context: SkillTurnContext


@dataclass(frozen=True)
class AIToolCall:
    """Llamada de herramienta devuelta por el proveedor de IA."""

    tool_call_id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class AIAssistantMessage:
    """Mensaje intermedio del assistant conservado durante el tool loop."""

    content: str | None
    tool_calls: tuple[AIToolCall, ...]
    reasoning_content: str | None


@dataclass(frozen=True)
class AIToolResult:
    """Resultado textual correlacionado con una llamada de herramienta."""

    tool_call_id: str
    content: str


@dataclass(frozen=True)
class SessionKey:
    """Identifica la sesión conversacional: privada o de grupo."""

    kind: ChatKind
    chat_id: int
    user_id: int | None

    def __post_init__(self) -> None:
        if self.kind not in (ChatKind.PRIVATE, ChatKind.GROUP):
            raise ValueError("Solo hay sesiones para chats privados o de grupo")
        if self.kind is ChatKind.PRIVATE and self.user_id is None:
            raise ValueError("Una sesión privada requiere user_id")

    @classmethod
    def private(cls, user_id: int) -> SessionKey:
        """Sesión de un chat privado (clave `private:<user_id>`)."""
        return cls(kind=ChatKind.PRIVATE, chat_id=user_id, user_id=user_id)

    @classmethod
    def group(cls, chat_id: int) -> SessionKey:
        """Sesión compartida de un grupo (clave `group:<chat_id>`)."""
        return cls(kind=ChatKind.GROUP, chat_id=chat_id, user_id=None)

    @property
    def value(self) -> str:
        """Representación estable de la clave (contrato de almacenamiento)."""
        if self.kind is ChatKind.PRIVATE:
            return f"private:{self.user_id}"
        return f"group:{self.chat_id}"


@dataclass(frozen=True)
class AttachmentKey:
    """Aísla el contexto temporal de adjuntos por chat, usuario y tipo de chat."""

    kind: ChatKind
    chat_id: int
    user_id: int

    def __post_init__(self) -> None:
        if self.kind not in (ChatKind.PRIVATE, ChatKind.GROUP):
            raise ValueError("Solo hay adjuntos para chats privados o de grupo")

    @property
    def value(self) -> str:
        """Representación estable y aislada para el almacenamiento temporal."""
        return f"{self.kind.value}:{self.chat_id}:{self.user_id}"


@dataclass(frozen=True)
class SessionState:
    """Historial de una sesión: turnos completos y última actividad."""

    messages: tuple[ChatMessage, ...]
    updated_at: datetime
    summary: str | None = None


@dataclass(frozen=True)
class IncomingMessage:
    """Mensaje entrante ya normalizado desde el adaptador de Telegram."""

    chat_id: int
    user_id: int
    chat_type: ChatKind
    text: str | None
    mentions_bot: bool
    replies_to_bot: bool
    quoted_text: str | None
    user_display_name: str | None = None


@dataclass(frozen=True)
class AppConfig:
    """Configuración de la aplicación cargada desde el entorno."""

    telegram_bot_token: str
    deepseek_api_key: str
    deepseek_model: str
    deepseek_timeout_s: float
    deepseek_max_tokens: int
    deepseek_system_prompt: str
    deepseek_thinking: bool
    deepseek_reasoning_effort: str
    allowed_user_ids: frozenset[int]
    allowed_chat_ids: frozenset[int]
    rejection_phrases: tuple[str, ...]
    only_text_phrases: tuple[str, ...]
    empty_mention_phrases: tuple[str, ...]
    session_max_turns: int
    session_idle_minutes: int
    new_session_phrases: tuple[str, ...]
    skills_enabled: bool
    skills_dir: str
    skills_autoload: bool
    skill_max_chars: int
    skills_catalog_max_chars: int
    skills_max_context_chars: int
    skills_max_per_turn: int
    skills_max_rounds: int
    skills_limit_phrase: str
    max_concurrent_updates: int
    ai_provider: str = "mimo"
    gpt_model: str = "gpt-6-luna"
    gpt_timeout_s: float = 30.0
    gpt_credential_path: str = ""
    mimo_api_key: str = ""
    mimo_model: str = "xiaomi/mimo-v2.6-flash"
    mimo_base_url: str = "https://api.xiaomimimo.com/v1"
    mimo_timeout_s: float = 30.0
    mimo_max_tokens: int = 1024
    workspace_enabled: bool = False
    workspace_dir: str = ""
    workspace_group_ids: frozenset[int] = frozenset()
    pi_enabled: bool = False
    pi_workspace_dir: str = ""
    pi_root_user_ids: frozenset[int] = frozenset()
    pi_stream_edit_interval_s: float = 1.0
