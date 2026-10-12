"""Modelos y puerto del dominio Pi.

Define los tipos para la integración con el agente Pi mediante RPC.
Ningún tipo incluye secretos ni credenciales.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class PiSessionState(Enum):
    """Estado de la máquina de estados del modo Pi por usuario."""

    NORMAL = "normal"
    PI_ACTIVO = "pi_activo"
    CONFIRMAR_SALIDA = "confirmar_salida"
    PI_ERROR = "pi_error"
    SELECCION_PROYECTO = "seleccion_proyecto"
    NOMBRE_PROYECTO = "nombre_proyecto"


@dataclass(frozen=True)
class PiProject:
    """Proyecto (subcarpeta) seleccionable como directorio de trabajo de Pi.

    `name` es el nombre visible (sin ruta); `path` es la ruta absoluta resuelta,
    garantizada dentro de la raíz de proyectos. `path` nunca se muestra al usuario.
    """

    name: str
    path: str


@dataclass(frozen=True)
class PiAccess:
    """Decisión de acceso al modo Pi para un mensaje entrante.

    `allowed` indica si el mensaje puede llegar a Pi. Si no, `reply` es el
    texto a responder (aviso o rechazo) o `None` para ignorarlo en silencio.
    """

    allowed: bool
    reply: str | None


@dataclass(frozen=True)
class PiEvent:
    """Evento emitido por Pi durante una ejecución.

    `type`: `text_delta`, `message_end`, `error`, `activity` (razonamiento u otro
    bloque, sin contenido), `settled` (Pi no seguirá trabajando) o `closed`
    (la sesión de Pi se cayó de forma inesperada).
    """

    type: str
    content: str
    tool_name: str | None = None


@dataclass(frozen=True)
class PiToolEvent:
    """Evento de una herramienta ejecutada por Pi: `tool_start`, `tool_update` o `tool_end`."""

    type: str
    tool_name: str


class PiModeError(Exception):
    """Fallo tipado del modo Pi con motivo interno y mensaje público opcional."""

    def __init__(self, reason: str, *, user_message: str | None = None) -> None:
        self.reason = reason
        self.user_message = user_message
        super().__init__(f"Error de modo Pi: {reason}")


class PiProjectStorePort(Protocol):
    """Puerto para listar, resolver y crear proyectos confinados a una raíz.

    `resolve` y `create` lanzan `PiModeError("project_invalid")` ante nombres
    inválidos o rutas fuera de la raíz; los fallos de E/S también se traducen a
    `PiModeError`.
    """

    def list_projects(self) -> list[PiProject]:
        """Devuelve los proyectos de primer nivel en orden estable."""
        ...

    def resolve(self, name: str) -> PiProject:
        """Resuelve un proyecto existente por nombre."""
        ...

    def create(self, name: str) -> tuple[PiProject, bool]:
        """Crea el proyecto o reutiliza uno existente; el bool indica si ya existía."""
        ...


@runtime_checkable
class PiAgentPort(Protocol):
    """Puerto de comunicación con un agente Pi mediante RPC."""

    async def send_prompt(
        self, text: str, images: list[bytes] | None = None
    ) -> None:
        """Envía un mensaje de usuario a Pi."""
        ...

    async def send_steer(
        self, text: str, images: list[bytes] | None = None
    ) -> None:
        """Envía una indicación de dirección mientras Pi trabaja."""
        ...

    async def abort(self) -> None:
        """Detiene el trabajo actual de Pi."""
        ...

    async def new_session(self) -> None:
        """Inicia una conversación nueva en Pi."""
        ...

    async def get_state(self) -> dict[str, object]:
        """Devuelve el estado actual de la sesión de Pi."""
        ...

    async def set_model(self, provider: str, model_id: str) -> dict[str, object]:
        """Cambia el modelo de la sesión de Pi y devuelve el modelo activo."""
        ...

    async def get_available_models(self) -> list[dict[str, object]]:
        """Devuelve los modelos disponibles en Pi."""
        ...

    async def get_commands(self) -> list[dict[str, object]]:
        """Devuelve los comandos descubribles de Pi (plantillas, skills, extensiones)."""
        ...

    async def bash(self, command: str) -> dict[str, object]:
        """Ejecuta un comando shell en el workspace de Pi y devuelve el resultado."""
        ...

    def on_event(self, callback: Callable[[PiEvent], None]) -> None:
        """Registra un callback para eventos de texto y error."""
        ...

    def on_tool_event(self, callback: Callable[[PiToolEvent], None]) -> None:
        """Registra un callback para eventos de herramientas."""
        ...

    async def close(self) -> None:
        """Cierra la conexión con Pi y libera recursos."""
        ...