"""Caso de uso: modo Pi — máquina de estados, streaming y comandos.

Gestiona la conversación directa con Pi desde Telegram, sin consultar
el LLM de Oraculo. Cada usuario tiene su propia sesión aislada.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass
from typing import Protocol

from oraculo.application.audio import format_voice_prompt, transcribe_voice
from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.errors import AudioTranscriptionError
from oraculo.domain.ports import AudioTranscriber
from oraculo.domain.pi import (
    PiAgentPort,
    PiEvent,
    PiModeError,
    PiProject,
    PiProjectStorePort,
    PiSessionState,
    PiToolEvent,
)

logger = logging.getLogger(__name__)

PI_PREFIX = "🤖💻"

TOOL_NOTICES: dict[str, str] = {
    "read": "Leyendo archivos",
    "write": "escribiendo archivos",
    "edit": "editando archivos",
    "bash": "Ejecutando comandos",
    "grep": "buscando en archivos",
    "find": "buscando archivos",
    "ls": "listando archivos",
}

TELEGRAM_MAX_CHARS = 4096
# Tiempo máximo sin ningún evento de Pi antes de abortar el turno (Req 3.3).
PI_IDLE_TIMEOUT_S = 300.0
CREATE_PROJECT_LABEL = "Crear un proyecto nuevo"
ROOT_PROJECT_LABEL = "Oraculo (raíz del proyecto)"

# Estados con Pi lanzado (hay agente). Los de selección aún no lanzan Pi.
_ACTIVE_STATES = (
    PiSessionState.PI_ACTIVO,
    PiSessionState.CONFIRMAR_SALIDA,
    PiSessionState.PI_ERROR,
)
_SELECTION_STATES = (
    PiSessionState.SELECCION_PROYECTO,
    PiSessionState.NOMBRE_PROYECTO,
)
_VOICE_STATES = (PiSessionState.PI_ACTIVO, PiSessionState.PI_ERROR)


def _model_field(model: dict[str, object], key: str) -> str:
    """Lee un campo de un modelo de Pi de forma defensiva, sin exponer secretos."""
    value = model.get(key, "")
    return value if isinstance(value, str) else ""


def is_pi_command(text: str) -> bool:
    """Detecta comandos propios del modo Pi (`/model`, `//…`) para enrutarlos."""
    return text == "/model" or text.startswith("/model ") or text.startswith("//")


def _model_line(model: dict[str, object]) -> str:
    return (
        f"{_model_field(model, 'provider')}/{_model_field(model, 'id')} "
        f"— {_model_field(model, 'name')}"
    )


@dataclass(frozen=True)
class PiDisplayEvent:
    """Evento de visualización para el adaptador de Telegram."""

    type: str
    content: str


class PiAgentFactory(Protocol):
    """Lanza Pi en `cwd`; `session_file` retoma una sesión guardada (Req 2.3)."""

    def __call__(
        self, user_id: int, cwd: str, session_file: str | None = None
    ) -> Awaitable[PiAgentPort]: ...


def _format_duration(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        return f"{minutes} minuto" + ("" if minutes == 1 else "s")
    return f"{seconds:g} segundos"


class PiModeService:
    """Gestiona el modo Pi por usuario: estados, comandos y streaming."""

    def __init__(
        self,
        *,
        agent_factory: PiAgentFactory,
        project_store: PiProjectStorePort,
        root_project: PiProject | None = None,
        root_user_ids: frozenset[int] = frozenset(),
        audio_transcriber: AudioTranscriber | None = None,
        idle_timeout_s: float = PI_IDLE_TIMEOUT_S,
    ) -> None:
        """`agent_factory(user_id, cwd[, session_file])` lanza Pi en la ruta del proyecto.

        `root_project` es la raíz de Oraculo como proyecto; solo los usuarios de
        `root_user_ids` la ven como opción adicional (raíz). `idle_timeout_s` es
        el tiempo máximo sin eventos de Pi antes de abortar un turno.
        """
        self._agent_factory = agent_factory
        self._project_store = project_store
        self._root_project = root_project
        self._root_user_ids = root_user_ids
        self._audio_transcriber = audio_transcriber
        self._idle_timeout_s = idle_timeout_s
        self._states: dict[int, PiSessionState] = {}
        self._agents: dict[int, PiAgentPort] = {}
        self._options: dict[int, list[PiProject]] = {}
        # Cola del turno en curso por usuario; sin turno, los eventos se descartan (Req 4.1).
        self._turns: dict[int, asyncio.Queue[PiEvent | PiToolEvent]] = {}
        # Proyecto de la sesión activa y si es la raíz; necesario para relanzar (Req 2.3).
        self._projects: dict[int, tuple[PiProject, bool]] = {}
        # Último `sessionFile` de Pi; nunca se muestra ni se registra (Req 2.3.3).
        self._session_files: dict[int, str] = {}
        self._crashed: set[int] = set()

    def _is_root(self, user_id: int) -> bool:
        """True si el usuario puede trabajar en la raíz del proyecto (Req 3.3)."""
        return self._root_project is not None and user_id in self._root_user_ids

    def get_state(self, user_id: int) -> PiSessionState:
        return self._states.get(user_id, PiSessionState.NORMAL)

    def is_in_pi_mode(self, user_id: int) -> bool:
        """Incluye la selección de proyecto: sus respuestas no deben ir al LLM (Req 5.1)."""
        state = self.get_state(user_id)
        return state in _ACTIVE_STATES or state in _SELECTION_STATES

    async def handle(
        self, user_id: int, text: str, *, images: list[bytes] | None = None
    ) -> AsyncIterator[PiDisplayEvent]:
        """Procesa un mensaje y emite eventos de visualización."""
        state = self.get_state(user_id)
        stripped = text.strip()

        if stripped == "/agent":
            async for event in self._enter_pi_mode(user_id, state):
                yield event
            return

        if state is PiSessionState.NORMAL:
            if is_pi_command(stripped):
                yield PiDisplayEvent(
                    type="response",
                    content=f"{PI_PREFIX} El modo agente no está activo. Usa /agent para activarlo.",
                )
            return

        if state in _SELECTION_STATES:
            if stripped == "/exitagent":
                self._reset_to_normal(user_id)
                yield PiDisplayEvent(
                    type="response",
                    content=f"{PI_PREFIX} Selección de proyecto cancelada. Modo agente desactivado.",
                )
            elif state is PiSessionState.SELECCION_PROYECTO:
                async for event in self._handle_selection(user_id, stripped):
                    yield event
            else:
                async for event in self._handle_project_name(user_id, stripped):
                    yield event
            return

        if stripped == "/exitagent":
            async for event in self._handle_exit(user_id, state):
                yield event
            return

        if state is PiSessionState.CONFIRMAR_SALIDA:
            async for event in self._handle_confirm(user_id, stripped):
                yield event
            return

        if stripped == "/stoppi":
            async for event in self._handle_stop(user_id):
                yield event
            return

        if stripped in ("/new", "/reset"):
            async for event in self._handle_reset(user_id):
                yield event
            return

        if stripped == "/agent":
            yield PiDisplayEvent(
                type="response",
                content=f"{PI_PREFIX} Ya estás en modo agente.",
            )
            return

        if stripped == "/model" or stripped.startswith("/model "):
            async for event in self._handle_model(user_id, stripped):
                yield event
            return

        if stripped.startswith("//"):
            async for event in self._handle_forward(user_id, stripped):
                yield event
            return

        async for event in self._handle_message(user_id, stripped, images=images):
            yield event

    async def handle_voice(
        self,
        user_id: int,
        attachment: IncomingAttachment,
        data: bytes | None,
        *,
        caption: str | None,
    ) -> AsyncIterator[PiDisplayEvent]:
        """Transcribe una nota de voz y la entrega como texto a Pi."""
        if self.get_state(user_id) not in _VOICE_STATES:
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} El agente no está listo para recibir una nota de voz.",
            )
            return

        try:
            transcript = await transcribe_voice(
                attachment, data, transcriber=self._audio_transcriber,
            )
        except AudioTranscriptionError as exc:
            logger.warning(
                "Pi: no se pudo transcribir nota de voz user_id=%s reason=%s",
                user_id,
                exc.reason,
            )
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} {self._voice_error_message(exc)}",
            )
            return

        if self.get_state(user_id) not in _VOICE_STATES:
            yield PiDisplayEvent(
                type="error",
                content=(
                    f"{PI_PREFIX} La sesión del agente cambió durante la transcripción; "
                    "no envié el texto."
                ),
            )
            return

        prompt = format_voice_prompt(caption, transcript)
        async for event in self._handle_message(user_id, prompt):
            yield event

    @staticmethod
    def _voice_error_message(error: AudioTranscriptionError) -> str:
        if error.reason == "file_too_large":
            return "La nota de voz supera el máximo de 20 MB permitido."
        if error.reason == "download_failed":
            return "No pude descargar la nota de voz; vuelve a enviarla."
        if error.reason == "unsupported_format":
            return "No pude reconocer el formato de la nota de voz."
        if error.reason in {"empty_audio", "empty_transcript"}:
            return "No detecté voz hablada en la nota de voz."
        if error.reason == "not_configured":
            return "El procesamiento de audio no está configurado en este bot."
        return "No pude transcribir la nota de voz; vuelve a intentarlo."

    async def _enter_pi_mode(
        self, user_id: int, state: PiSessionState
    ) -> AsyncIterator[PiDisplayEvent]:
        """`/agent`: lista proyectos sin lanzar Pi (Req 1.1); re-lista en selección (Req 5.3)."""
        if state in _ACTIVE_STATES:
            yield PiDisplayEvent(
                type="response",
                content=f"{PI_PREFIX} Ya estás en modo agente.",
            )
            return
        try:
            projects = self._project_store.list_projects()
        except PiModeError as exc:
            self._reset_to_normal(user_id)
            msg = exc.user_message or "No se pudo leer la carpeta de proyectos."
            yield PiDisplayEvent(type="error", content=f"{PI_PREFIX} {msg}")
            return
        self._options[user_id] = projects
        self._states[user_id] = PiSessionState.SELECCION_PROYECTO
        yield PiDisplayEvent(
            type="response",
            content=self._format_projects(projects, include_root=self._is_root(user_id)),
        )

    @staticmethod
    def _format_projects(projects: list[PiProject], *, include_root: bool = False) -> str:
        """Lista numerada estable; «crear nuevo» y, para root, la raíz al final.

        (Req 1.2, 1.4 spec base; Req 2.1, 3.1 spec root.)
        """
        if projects:
            header = (
                f"{PI_PREFIX} Elige el proyecto en el que "
                "trabajarás en modo agente (responde con el número):"
            )
        else:
            header = (
                f"{PI_PREFIX} No hay proyectos en la carpeta de trabajo. "
                "Responde con el número para crear uno:"
            )
        lines = [f"{i}. {p.name}" for i, p in enumerate(projects, start=1)]
        lines.append(f"{len(projects) + 1}. {CREATE_PROJECT_LABEL}")
        if include_root:
            lines.append(f"{len(projects) + 2}. {ROOT_PROJECT_LABEL}")
        return "\n".join([header, *lines, "Usa /exitagent para cancelar."])

    def _selection_retry(self, user_id: int, message: str) -> list[PiDisplayEvent]:
        """Error + lista vigente para volver a elegir (Req 2.2)."""
        return [
            PiDisplayEvent(type="error", content=f"{PI_PREFIX} {message}"),
            PiDisplayEvent(
                type="response",
                content=self._format_projects(
                    self._options.get(user_id, []), include_root=self._is_root(user_id)
                ),
            ),
        ]

    async def _handle_selection(
        self, user_id: int, text: str
    ) -> AsyncIterator[PiDisplayEvent]:
        options = self._options.get(user_id, [])
        create_index = len(options) + 1
        is_root = self._is_root(user_id)
        root_index = create_index + 1 if is_root else None
        max_index = root_index if root_index is not None else create_index
        choice = int(text) if text.isascii() and text.isdigit() else 0
        if not 1 <= choice <= max_index:
            for event in self._selection_retry(
                user_id, f"Opción no válida. Responde con un número del 1 al {max_index}."
            ):
                yield event
            return
        if choice == root_index:
            # `self._root_project` no es None: `is_root` lo garantiza (Req 2.2).
            assert self._root_project is not None
            async for event in self._launch(
                user_id, self._root_project, reused=False, is_root=True
            ):
                yield event
            return
        if choice == create_index:
            self._states[user_id] = PiSessionState.NOMBRE_PROYECTO
            yield PiDisplayEvent(
                type="response",
                content=(
                    f"{PI_PREFIX} Escribe el nombre del proyecto nuevo "
                    "(letras, números, «-», «_» o «.»). Usa /exitagent para cancelar."
                ),
            )
            return
        try:
            # Revalida en disco: la carpeta pudo cambiar tras listar (Req 4.1).
            project = self._project_store.resolve(options[choice - 1].name)
        except PiModeError as exc:
            for event in self._selection_retry(
                user_id, exc.user_message or "El proyecto no está disponible."
            ):
                yield event
            return
        async for event in self._launch(user_id, project, reused=False):
            yield event

    async def _handle_project_name(
        self, user_id: int, text: str
    ) -> AsyncIterator[PiDisplayEvent]:
        try:
            project, existed = self._project_store.create(text)
        except PiModeError as exc:
            # Nombre inválido o fallo al crear: se vuelve a pedir (Req 3.4, 6.2).
            msg = exc.user_message or "No se pudo crear el proyecto."
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} {msg} Escribe otro nombre o usa /exitagent para cancelar.",
            )
            return
        async for event in self._launch(user_id, project, reused=existed):
            yield event

    async def _launch(
        self, user_id: int, project: PiProject, *, reused: bool, is_root: bool = False
    ) -> AsyncIterator[PiDisplayEvent]:
        """Lanza Pi con el proyecto como `cwd`; ante fallo permite reintentar (Req 6.3)."""
        try:
            agent = await self._agent_factory(user_id, project.path)
        except PiModeError as exc:
            self._states[user_id] = PiSessionState.SELECCION_PROYECTO
            try:
                self._options[user_id] = self._project_store.list_projects()
            except PiModeError:
                logger.warning("Pi: no se pudo refrescar la lista de proyectos")
            for event in self._selection_retry(user_id, exc.user_message or "No se pudo iniciar el agente."):
                yield event
            return
        self._options.pop(user_id, None)
        self._session_files.pop(user_id, None)
        self._install_agent(user_id, agent, project, is_root=is_root)
        if is_root:
            # No se muestra la ruta absoluta (Req 2.3 de `modo-pi-root`).
            yield PiDisplayEvent(
                type="response",
                content=(
                    f"{PI_PREFIX} Modo agente activado en la raíz del proyecto Oraculo. "
                    "Escribe tu primera instrucción."
                ),
            )
            return
        notice = f" El proyecto «{project.name}» ya existía; se reutiliza." if reused else ""
        yield PiDisplayEvent(
            type="response",
            content=(
                f"{PI_PREFIX} Modo agente activado en el proyecto «{project.name}».{notice} "
                "Escribe tu primera instrucción."
            ),
        )

    def _install_agent(
        self, user_id: int, agent: PiAgentPort, project: PiProject, *, is_root: bool
    ) -> None:
        """Activa el agente y registra sus callbacks una sola vez (Req 4.1, 4.2)."""
        self._agents[user_id] = agent
        self._projects[user_id] = (project, is_root)
        self._crashed.discard(user_id)
        self._states[user_id] = PiSessionState.PI_ACTIVO
        if is_root:
            # Se registra el acceso a la raíz (Req 4.1 de `modo-pi-root`).
            logger.info("Pi: sesión en la raíz del proyecto user_id=%s", user_id)

        def on_event(event: PiEvent) -> None:
            self._route(user_id, agent, event)

        def on_tool_event(event: PiToolEvent) -> None:
            self._route(user_id, agent, event)

        agent.on_event(on_event)
        agent.on_tool_event(on_tool_event)

    def _route(
        self, user_id: int, agent: PiAgentPort, event: PiEvent | PiToolEvent
    ) -> None:
        """Entrega el evento solo al turno en curso del agente vigente (Req 4.1)."""
        if isinstance(event, PiEvent) and event.type == "closed" and self._agents.get(user_id) is agent:
            self._crashed.add(user_id)
        if self._agents.get(user_id) is not agent:
            return
        queue = self._turns.get(user_id)
        if queue is not None:
            queue.put_nowait(event)

    def _remember_session(self, user_id: int, state: dict[str, object]) -> None:
        session_file = state.get("sessionFile")
        if isinstance(session_file, str) and session_file:
            self._session_files[user_id] = session_file

    def _reset_to_normal(self, user_id: int) -> None:
        self._options.pop(user_id, None)
        self._states[user_id] = PiSessionState.NORMAL

    async def _handle_exit(
        self, user_id: int, state: PiSessionState
    ) -> AsyncIterator[PiDisplayEvent]:
        agent = self._agents.get(user_id)
        if agent is not None:
            try:
                agent_state = await agent.get_state()
                is_streaming = agent_state.get("isStreaming", False)
            except Exception:
                is_streaming = False
            if is_streaming:
                self._states[user_id] = PiSessionState.CONFIRMAR_SALIDA
                yield PiDisplayEvent(
                    type="response",
                    content=(
                        f"{PI_PREFIX} El agente todavía está trabajando. "
                        "¿Deseas detenerlo y salir? "
                        "Responde «sí» para salir o «no» para continuar."
                    ),
                )
                return
        await self._cleanup_user(user_id)
        self._forget_session(user_id)
        self._states[user_id] = PiSessionState.NORMAL
        yield PiDisplayEvent(
            type="response",
            content=f"{PI_PREFIX} Modo agente desactivado.",
        )

    async def _handle_confirm(
        self, user_id: int, text: str
    ) -> AsyncIterator[PiDisplayEvent]:
        if text.lower() in ("sí", "si", "yes"):
            await self._cleanup_user(user_id)
            self._forget_session(user_id)
            self._states[user_id] = PiSessionState.NORMAL
            yield PiDisplayEvent(
                type="response",
                content=f"{PI_PREFIX} Trabajo detenido. Modo agente desactivado.",
            )
        else:
            self._states[user_id] = PiSessionState.PI_ACTIVO
            yield PiDisplayEvent(
                type="response",
                content=f"{PI_PREFIX} Continúas en modo agente.",
            )

    async def _handle_stop(
        self, user_id: int
    ) -> AsyncIterator[PiDisplayEvent]:
        agent = self._agents.get(user_id)
        if agent is not None:
            try:
                await agent.abort()
            except Exception:
                pass
        yield PiDisplayEvent(
            type="response",
            content=f"{PI_PREFIX} Trabajo actual detenido.",
        )

    async def _handle_reset(
        self, user_id: int
    ) -> AsyncIterator[PiDisplayEvent]:
        agent = self._agents.get(user_id)
        if agent is not None:
            try:
                await agent.new_session()
            except Exception:
                pass
        yield PiDisplayEvent(
            type="response",
            content=f"{PI_PREFIX} Conversación reiniciada.",
        )

    async def _handle_model(
        self, user_id: int, text: str
    ) -> AsyncIterator[PiDisplayEvent]:
        agent = self._agents.get(user_id)
        if agent is None:
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} No hay una sesión de agente activa.",
            )
            return
        arg = text[len("/model"):].strip()
        try:
            models = await agent.get_available_models()
            if not arg:
                state = await agent.get_state()
                active = state.get("model")
                yield PiDisplayEvent(
                    type="response",
                    content=self._format_models(models, active),
                )
                return
            candidates, match = self._match_model(models, arg)
            if match is not None:
                provider = _model_field(match, "provider")
                model_id = _model_field(match, "id")
                updated = await agent.set_model(provider, model_id)
                shown = updated if updated else match
                yield PiDisplayEvent(
                    type="response",
                    content=f"{PI_PREFIX} Modelo cambiado a {_model_line(shown)}.",
                )
            elif candidates:
                lines = [_model_line(m) for m in candidates]
                yield PiDisplayEvent(
                    type="response",
                    content=(
                        f"{PI_PREFIX} El patrón coincide con varios modelos:\n"
                        + "\n".join(lines)
                        + "\nUsa un patrón más específico."
                    ),
                )
            else:
                yield PiDisplayEvent(
                    type="error",
                    content=f"{PI_PREFIX} No hay modelos que coincidan con «{arg}».",
                )
        except PiModeError as exc:
            msg = exc.user_message or "No se pudo consultar los modelos."
            yield PiDisplayEvent(type="error", content=f"{PI_PREFIX} {msg}")

    @staticmethod
    def _match_model(
        models: list[dict[str, object]], pattern: str
    ) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        """Coincidencia exacta por `provider/id` o `id`; si no, subcadena sobre id/name."""
        norm = pattern.strip().lower()
        exact = [
            m
            for m in models
            if _model_field(m, "id").lower() == norm
            or f"{_model_field(m, 'provider')}/{_model_field(m, 'id')}".lower() == norm
        ]
        if len(exact) == 1:
            return exact, exact[0]
        if len(exact) > 1:
            return exact, None
        fuzzy = [
            m
            for m in models
            if norm in _model_field(m, "id").lower()
            or norm in _model_field(m, "name").lower()
        ]
        return fuzzy, (fuzzy[0] if len(fuzzy) == 1 else None)

    @staticmethod
    def _format_models(models: list[dict[str, object]], active: object) -> str:
        if isinstance(active, dict) and _model_field(active, "id"):
            active_line = _model_line(active)
        else:
            active_line = "ninguno"
        lines = [_model_line(m) for m in models]
        listing = "\n".join(lines) if lines else "(sin modelos disponibles)"
        return (
            f"{PI_PREFIX} Modelo activo: {active_line}\n"
            f"{PI_PREFIX} Modelos disponibles:\n{listing}"
        )[:TELEGRAM_MAX_CHARS]

    async def _handle_forward(
        self, user_id: int, text: str
    ) -> AsyncIterator[PiDisplayEvent]:
        agent = self._agents.get(user_id)
        if agent is None:
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} No hay una sesión de agente activa.",
            )
            return
        body = text[2:].strip()
        if not body:
            yield PiDisplayEvent(
                type="response",
                content=f"{PI_PREFIX} Uso: //sh <comando> o //<comando> [args]",
            )
            return
        if body == "sh" or body.startswith("sh "):
            command = body[2:].strip()
            if not command:
                yield PiDisplayEvent(
                    type="response",
                    content=f"{PI_PREFIX} Uso: //sh <comando>",
                )
                return
            try:
                result = await agent.bash(command)
            except PiModeError as exc:
                msg = exc.user_message or "No se pudo ejecutar el comando."
                yield PiDisplayEvent(type="error", content=f"{PI_PREFIX} {msg}")
                return
            output = str(result.get("output", ""))
            exit_code = result.get("exitCode")
            truncated = bool(result.get("truncated", False))
            content = f"{PI_PREFIX} $ {command}\n{output}\n[código de salida: {exit_code}]"
            if truncated:
                content += " (salida truncada)"
            yield PiDisplayEvent(type="response", content=content[:TELEGRAM_MAX_CHARS])
            return
        name, _, args = body.partition(" ")
        name = name.strip()
        args = args.strip()
        try:
            commands = await agent.get_commands()
        except PiModeError as exc:
            msg = exc.user_message or "No se pudieron listar los comandos."
            yield PiDisplayEvent(type="error", content=f"{PI_PREFIX} {msg}")
            return
        match = next(
            (c for c in commands if str(c.get("name", "")).lower() == name.lower()),
            None,
        )
        if match is None:
            available = ", ".join(
                sorted(str(c.get("name", "")) for c in commands if c.get("name"))
            ) or "ninguno"
            yield PiDisplayEvent(
                type="error",
                content=(
                    f"{PI_PREFIX} Comando desconocido «{name}». "
                    f"Disponibles: {available}"
                ),
            )
            return
        prompt_text = f"/{name}" + (f" {args}" if args else "")
        async for event in self._handle_message(user_id, prompt_text):
            yield event

    async def _handle_message(
        self, user_id: int, text: str, *, images: list[bytes] | None = None
    ) -> AsyncIterator[PiDisplayEvent]:
        if user_id in self._crashed:
            relaunched = False
            async for event in self._relaunch(user_id):
                relaunched = relaunched or event.type == "response"
                yield event
            if not relaunched:
                return

        agent = self._agents.get(user_id)
        if agent is None:
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} No hay una sesión de agente activa.",
            )
            return

        queue: asyncio.Queue[PiEvent | PiToolEvent] = asyncio.Queue()
        self._turns[user_id] = queue
        try:
            try:
                agent_state = await agent.get_state()
                self._remember_session(user_id, agent_state)
                is_streaming = agent_state.get("isStreaming", False)
                logger.info("Pi: enviando mensaje streaming=%s", is_streaming)
                if is_streaming:
                    await agent.send_steer(text, images=images)
                    yield PiDisplayEvent(
                        type="response",
                        content=f"{PI_PREFIX} Mensaje enviado como indicación al agente.",
                    )
                    return
                await agent.send_prompt(text, images=images)
            except PiModeError as exc:
                if exc.reason == "process_crashed":
                    self._crashed.add(user_id)
                msg = exc.user_message or "Error al enviar el mensaje al agente."
                yield PiDisplayEvent(type="error", content=f"{PI_PREFIX} {msg}")
                return

            async for event in self._consume_turn(user_id, agent, queue):
                yield event
        finally:
            # Tras el turno (éxito, error, timeout o cancelación) no recibe más eventos.
            if self._turns.get(user_id) is queue:
                del self._turns[user_id]

    async def _consume_turn(
        self,
        user_id: int,
        agent: PiAgentPort,
        queue: asyncio.Queue[PiEvent | PiToolEvent],
    ) -> AsyncIterator[PiDisplayEvent]:
        """Traduce eventos hasta `settled`, `error`, `closed` o inactividad (Req 3.x, 2.1)."""
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=self._idle_timeout_s)
            except asyncio.TimeoutError:
                logger.warning("Pi: turno sin eventos; se aborta user_id=%s", user_id)
                try:
                    await agent.abort()
                except PiModeError as exc:
                    logger.warning("Pi: no se pudo abortar tras timeout reason=%s", exc.reason)
                yield PiDisplayEvent(
                    type="error",
                    content=(
                        f"{PI_PREFIX} El agente no respondió en "
                        f"{_format_duration(self._idle_timeout_s)}; detuve el trabajo."
                    ),
                )
                return
            if isinstance(event, PiToolEvent):
                if event.type == "tool_start":
                    notice = TOOL_NOTICES.get(event.tool_name, "trabajando")
                    if event.tool_name in {"read", "bash"}:
                        content = f"{PI_PREFIX} {notice}…"
                    else:
                        content = f"{PI_PREFIX} El agente está {notice}…"
                    yield PiDisplayEvent(type="tool_notice", content=content)
                continue
            if event.type == "text_delta":
                yield PiDisplayEvent(type="text_delta", content=event.content)
            elif event.type == "message_end":
                # Un turno puede tener varios mensajes de asistente (Req 3.2).
                yield PiDisplayEvent(type="text_final", content=event.content)
            elif event.type == "error":
                yield PiDisplayEvent(type="error", content=f"{PI_PREFIX} {event.content}")
                return
            elif event.type == "closed":
                yield PiDisplayEvent(
                    type="error",
                    content=(
                        f"{PI_PREFIX} La sesión del agente se cerró inesperadamente. "
                        "Escribe de nuevo para reanudarla."
                    ),
                )
                return
            elif event.type == "settled":
                return
            # `activity` y demás: solo reinician el temporizador de inactividad.

    def _project_label(self, project: PiProject, is_root: bool) -> str:
        return ROOT_PROJECT_LABEL if is_root else f"«{project.name}»"

    async def _start_agent(
        self, user_id: int, project: PiProject, session_file: str | None
    ) -> PiAgentPort | None:
        """Lanza Pi y comprueba que responde; devuelve None si falla (sin registrar rutas)."""
        mode = "reanudación" if session_file else "sesión nueva"
        try:
            if session_file:
                agent = await self._agent_factory(
                    user_id, project.path, session_file=session_file
                )
            else:
                agent = await self._agent_factory(user_id, project.path)
        except PiModeError as exc:
            logger.warning(
                "Pi: relanzamiento fallido modo=%s reason=%s user_id=%s",
                mode, exc.reason, user_id,
            )
            return None
        try:
            await agent.get_state()
        except PiModeError as exc:
            logger.warning(
                "Pi: Pi no respondió tras relanzar modo=%s reason=%s user_id=%s",
                mode, exc.reason, user_id,
            )
            await self._close_agent(agent)
            return None
        logger.info("Pi: relanzamiento ok modo=%s user_id=%s", mode, user_id)
        return agent

    async def _relaunch(self, user_id: int) -> AsyncIterator[PiDisplayEvent]:
        """Relanza tras una caída: reanuda, si no sesión limpia, si no PI_ERROR (Req 2.3)."""
        entry = self._projects.get(user_id)
        await self._cleanup_user(user_id)
        if entry is None:
            self._crashed.discard(user_id)
            self._states[user_id] = PiSessionState.PI_ERROR
            yield PiDisplayEvent(
                type="error",
                content=f"{PI_PREFIX} No se pudo reiniciar el agente. Usa /exitagent y /agent.",
            )
            return
        project, is_root = entry
        label = self._project_label(project, is_root)
        session_file = self._session_files.get(user_id)
        if session_file:
            agent = await self._start_agent(user_id, project, session_file)
            if agent is not None:
                self._install_agent(user_id, agent, project, is_root=is_root)
                yield PiDisplayEvent(
                    type="response",
                    content=(
                        f"{PI_PREFIX} La sesión del agente se había cerrado; "
                        f"la reanudé en {label}."
                    ),
                )
                return
        agent = await self._start_agent(user_id, project, None)
        if agent is not None:
            self._session_files.pop(user_id, None)
            self._install_agent(user_id, agent, project, is_root=is_root)
            yield PiDisplayEvent(
                type="response",
                content=(
                    f"{PI_PREFIX} La sesión del agente se había cerrado y no pude reanudarla; "
                    f"inicié una nueva en {label}. La conversación anterior se perdió."
                ),
            )
            return
        # Sigue en modo Pi (PI_ERROR ∈ _ACTIVE_STATES) y reintenta con el próximo mensaje.
        self._crashed.add(user_id)
        self._states[user_id] = PiSessionState.PI_ERROR
        yield PiDisplayEvent(
            type="error",
            content=(
                f"{PI_PREFIX} No se pudo reiniciar el agente. "
                "Escribe de nuevo para reintentar o usa /exitagent."
            ),
        )

    @staticmethod
    async def _close_agent(agent: PiAgentPort) -> None:
        try:
            await agent.close()
        except PiModeError as exc:
            logger.warning("Pi: fallo al cerrar el agente reason=%s", exc.reason)

    async def _cleanup_user(self, user_id: int) -> None:
        self._turns.pop(user_id, None)
        agent = self._agents.pop(user_id, None)
        if agent is not None:
            await self._close_agent(agent)

    def _forget_session(self, user_id: int) -> None:
        """Olvida proyecto, sesión y caída del usuario al salir del modo Pi."""
        self._projects.pop(user_id, None)
        self._session_files.pop(user_id, None)
        self._crashed.discard(user_id)