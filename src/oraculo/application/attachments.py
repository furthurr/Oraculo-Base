"""Caso de uso: aceptar, clasificar y retener archivos adjuntos por sesión."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath

from oraculo.domain.attachments import (
    AttachmentMode,
    AttachmentPayload,
    IncomingAttachment,
    StoredAttachment,
)
from oraculo.domain.errors import AttachmentError
from oraculo.domain.models import AttachmentKey, ChatKind, IncomingMessage
from oraculo.domain.ports import AttachmentStore

MAX_FILES_PER_INTERACTION = 5
MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB, techo de descarga de Telegram.

logger = logging.getLogger(__name__)

_IMAGE_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\xff\xd8", "image/jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
    (b"GIF87a", "image/gif", ".gif"),
    (b"GIF89a", "image/gif", ".gif"),
)
_MIME_PATTERN = re.compile(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+")
_AUDIO_AAC_MIME_TYPES = frozenset({"audio/aac", "audio/x-aac"})
_AUDIO_MP4_MIME_TYPES = frozenset({"audio/mp4", "audio/x-m4a", "audio/m4a"})

_NEGATIVE = ("no", "no procedas", "espera", "aún no", "aun no", "cancela",
             "otros", "otro", "otros archivos", "reemplaza")
_CONFIRMATION = ("procede", "procesa", "sí", "si", "ok", "vale", "continúa",
                 "continua", "adelante", "de acuerdo", "hazlo", "sigue")


def _is_negative(text: str) -> bool:
    lowered = text.lower().strip(" .!¡")
    return lowered in _NEGATIVE or lowered.startswith("no ") or lowered.startswith("no,")


def _is_confirmation(text: str) -> bool:
    lowered = text.lower().strip(" .!¡")
    return lowered in _CONFIRMATION


def is_stale(updated_at: datetime, now: datetime, *, minutes: int) -> bool:
    """Indica si una actividad quedó caducada por inactividad."""
    if minutes <= 0:
        return False
    from datetime import timezone

    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now - updated_at).total_seconds() / 60 > minutes


def attachments_allowed(
    message: IncomingMessage,
    *,
    allowed_users: frozenset[int],
    allowed_chats: frozenset[int],
) -> bool:
    """Aplica a los adjuntos el mismo control de acceso que al texto."""
    if message.chat_type is ChatKind.PRIVATE:
        return message.user_id in allowed_users
    if message.chat_type is ChatKind.GROUP:
        return message.chat_id in allowed_chats and (
            message.mentions_bot or message.replies_to_bot
        )
    return False


class NextAction(Enum):
    """Qué debe hacer el caller tras un `intake`."""

    PROCESS = "process"                # procesar con el LLM (hay intención)
    ASK_INTENT = "ask_intent"          # preguntar: válidos u otros
    ASK_INSTRUCTIONS = "ask_instructions"  # pedir qué hacer con los archivos


@dataclass(frozen=True)
class IntakeOutcome:
    """Resultado de aceptar una interacción de adjuntos."""

    accepted: tuple[StoredAttachment, ...]
    notices: tuple[str, ...]
    action: NextAction
    rejected: tuple[str, ...] = ()
    has_valid_attachments: bool = False


@dataclass(frozen=True)
class IntentDecision:
    """Resultado de resolver la intención del usuario."""

    action: NextAction
    request: str | None
    notices: tuple[str, ...] = ()


@dataclass
class IntakeState:
    """Estado temporal de la sesión mientras se decide la intención."""

    valid: list[StoredAttachment] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    awaiting_intent: bool = False
    last_request: str | None = None
    updated_at: datetime | None = None


class AttachmentService:
    """Clasifica y retiene adjuntos temporalmente para un usuario y chat."""

    def __init__(
        self,
        *,
        store: AttachmentStore,
        binary_supported: Callable[[str, str], bool],
        clock: Callable[[], datetime],
        extract_text: Callable[[str, bytes], str] | None = None,
        max_files: int = MAX_FILES_PER_INTERACTION,
        max_file_bytes: int = MAX_FILE_BYTES,
        idle_minutes: int = 240,
    ) -> None:
        self._store = store
        self._binary_supported = binary_supported
        self._clock = clock
        self._extract_text = extract_text
        self._max_files = max_files
        self._max_file_bytes = max_file_bytes
        self._idle_minutes = idle_minutes
        self._states: dict[str, IntakeState] = {}

    def _state(self, key: AttachmentKey) -> IntakeState:
        state = self._states.get(key.value)
        if state is None:
            state = IntakeState()
            self._states[key.value] = state
        return state

    async def intake(
        self,
        key: AttachmentKey,
        items: Sequence[tuple[IncomingAttachment, bytes | None]],
        *,
        request_text: str | None,
    ) -> IntakeOutcome:
        """Acepta una interacción de adjuntos y decide el siguiente paso.

        Cada elemento de `items` es el adjunto normalizado y sus bytes ya
        descargados (`None` si la descarga falló).
        """
        state = self._state(key)
        had_valid_attachments = bool(state.valid)
        notices: list[str] = []
        rejected: list[str] = []
        accepted: list[StoredAttachment] = []

        available_slots = max(0, self._max_files - len(state.valid))
        selected = list(items[:available_slots])
        if len(items) > available_slots:
            notices.append(
                f"El máximo es {self._max_files} archivos por solicitud; usa "
                "/reset para empezar otra y adjuntar el resto."
            )

        for attachment, data in selected:
            stored = await self._accept(key, attachment, data, notices, rejected)
            if stored is not None:
                accepted.append(stored)

        state.valid.extend(accepted)
        state.rejected.extend(rejected)
        normalized_request = request_text.strip() if request_text and request_text.strip() else None
        if normalized_request is not None:
            state.last_request = normalized_request
        elif not had_valid_attachments:
            state.last_request = None
        state.awaiting_intent = bool(state.valid) and (
            bool(rejected) or normalized_request is None
        )
        state.updated_at = self._clock()

        if rejected:
            action = NextAction.ASK_INTENT
        elif normalized_request is None:
            action = NextAction.ASK_INSTRUCTIONS
        else:
            action = NextAction.PROCESS

        logger.info(
            "Adjuntos de la interacción key=%s aceptados=%s rechazados=%s accion=%s",
            key.value, len(accepted), len(rejected), action.value,
        )

        return IntakeOutcome(
            accepted=tuple(accepted),
            notices=tuple(notices),
            action=action,
            rejected=tuple(rejected),
            has_valid_attachments=bool(state.valid),
        )

    async def resolve_intent(self, key: AttachmentKey, text: str) -> IntentDecision:
        """Resuelve la respuesta del usuario ante la pregunta de intención.

        Negaciones mantienen la espera; cualquier otra respuesta confirma el
        procesado con los archivos válidos. Si aún no hay solicitud, el texto
        del usuario la aporta.
        """
        state = self._states.get(key.value)
        if state is None:
            return IntentDecision(
                action=NextAction.ASK_INSTRUCTIONS,
                request=None,
                notices=(
                    "No hay archivos compatibles pendientes; adjunta uno compatible "
                    "o reinicia la solicitud.",
                ),
            )
        state.updated_at = self._clock()
        stripped = text.strip()
        if not state.valid:
            state.awaiting_intent = False
            return IntentDecision(
                action=NextAction.ASK_INSTRUCTIONS,
                request=None,
                notices=(
                    "No hay archivos compatibles pendientes; adjunta uno compatible "
                    "o reinicia la solicitud.",
                ),
            )
        if _is_negative(stripped):
            state.awaiting_intent = True
            return IntentDecision(
                action=NextAction.ASK_INTENT,
                request=state.last_request,
                notices=("Vale, espero; dime cuándo proceder o adjunta otros archivos.",),
            )
        if not state.valid and not state.last_request:
            # Confirmación u opinión huérfana: no hay nada que procesar.
            return IntentDecision(
                action=NextAction.ASK_INSTRUCTIONS,
                request=None,
                notices=("¿Qué quieres hacer con los archivos?",),
            )
        request = stripped or state.last_request or None
        if _is_confirmation(stripped):
            request = state.last_request or stripped or None
        state.awaiting_intent = False
        if not request:
            return IntentDecision(
                action=NextAction.ASK_INSTRUCTIONS,
                request=None,
                notices=("¿Qué quieres hacer con los archivos?",),
            )
        return IntentDecision(action=NextAction.PROCESS, request=request)

    async def payloads(self, key: AttachmentKey) -> tuple[AttachmentPayload, ...]:
        """Contenido de los adjuntos vigentes listo para enviar al LLM."""
        state = self._states.get(key.value)
        if state is None:
            return ()
        state.updated_at = self._clock()
        payloads: list[AttachmentPayload] = []
        for stored in state.valid:
            data = await self._store.load(key, stored.name)
            if stored.mode is AttachmentMode.BINARY:
                payloads.append(AttachmentPayload(
                    name=stored.name, mime_type=stored.mime_type,
                    mode=AttachmentMode.BINARY, data=data,
                ))
            else:
                text = self._extract_text(stored.name, data) if self._extract_text else ""
                payloads.append(AttachmentPayload(
                    name=stored.name, mime_type=stored.mime_type,
                    mode=AttachmentMode.TEXT, text=text,
                ))
        return tuple(payloads)

    async def reset(self, key: AttachmentKey) -> None:
        """Elimina los adjuntos y el estado temporal del contexto indicado."""
        self._states.pop(key.value, None)
        await self._store.clear(key)

    async def expire_if_stale(self, key: AttachmentKey) -> bool:
        """Borra los adjuntos si el contexto caducó por inactividad.

        Usa el plazo configurado para las sesiones; cada uso renueva actividad.
        """
        state = self._states.get(key.value)
        if state is None:
            return False
        if state.updated_at is not None and is_stale(
            state.updated_at, self._clock(), minutes=self._idle_minutes
        ):
            logger.info(
                "Adjuntos caducados key=%s idle_minutes=%s",
                key.value, self._idle_minutes,
            )
            await self.reset(key)
            return True
        return False

    def is_awaiting(self, key: AttachmentKey) -> bool:
        """Indica si la sesión espera una decisión de intención."""
        state = self._states.get(key.value)
        return state.awaiting_intent if state is not None else False

    async def startup(self) -> None:
        """Limpieza de arranque: ninguna sesión sobrevive al proceso."""
        self._states.clear()
        await self._store.clear_all()

    async def _accept(
        self,
        key: AttachmentKey,
        attachment: IncomingAttachment,
        data: bytes | None,
        notices: list[str],
        rejected: list[str],
    ) -> StoredAttachment | None:
        mime_type = (attachment.mime_type or "").split(";", 1)[0].strip().lower()
        if mime_type == "application/octet-stream" or not _MIME_PATTERN.fullmatch(mime_type):
            mime_type = ""
        audio_candidate = attachment.kind in {"audio", "voice"} or mime_type.startswith("audio/")
        if attachment.kind == "voice":
            name = "nota de voz"
        elif audio_candidate:
            name = "audio adjunto"
        else:
            name = attachment.file_name or (
                "archivo adjunto" if attachment.kind == "document"
                else f"archivo-{attachment.file_id}"
            )
        size = attachment.file_size if attachment.file_size is not None else 0

        if size > self._max_file_bytes or (data is not None and len(data) > self._max_file_bytes):
            notices.append(
                f"«{name}» supera el peso máximo de 20 MB; no se puede subir "
                "al bot archivos mayores de ese tamaño."
            )
            rejected.append(name)
            return None

        if data is None:
            notices.append(f"No pude descargar «{name}»; lo omito y sigo con el resto.")
            rejected.append(name)
            return None

        detected_image = self._detect_image(data)
        audio_format = detect_audio_format(
            data, mime_type=mime_type, kind=attachment.kind,
        )
        if attachment.kind == "voice":
            notices.append("Las notas de voz se procesan como mensajes, no como adjuntos pendientes.")
            rejected.append(name)
            return None
        if audio_format is not None:
            mime_type, extension = audio_format
            name = await self._next_audio_name(key, extension)
            stored = StoredAttachment(
                name=name, mime_type=mime_type, mode=AttachmentMode.BINARY,
                size=len(data),
            )
            await self._store.store(key, stored.name, stored.mime_type, stored.mode, data)
            return stored
        if audio_candidate:
            notices.append(
                f"No puedo procesar «{name}» porque su formato de audio no está soportado."
            )
            rejected.append(name)
            return None

        excluded_media = (
            attachment.kind == "video"
            or mime_type.startswith("video/")
            or (attachment.kind == "animation" and detected_image is None)
        )
        if excluded_media:
            mime_type = ""
        elif detected_image is not None:
            detected_mime, extension = detected_image
            # El contenido prevalece sobre un MIME de imagen ausente o
            # contradictorio; PhotoSize no incluye nombre ni MIME.
            mime_type = detected_mime
            if not attachment.file_name:
                name = f"archivo-{attachment.file_id}{extension}"
            else:
                original = PurePosixPath(attachment.file_name.replace("\\", "/"))
                if original.suffix.lower() != extension:
                    name = f"{original.stem if original.suffix else original.name}{extension}"
        elif attachment.kind == "photo" or mime_type.startswith("image/"):
            mime_type = ""

        if mime_type and self._binary_supported(name, mime_type):
            mode = AttachmentMode.BINARY
        elif (
            not excluded_media
            and attachment.kind != "photo"
            and not (attachment.mime_type or "").lower().startswith("image/")
            and self._extractable(name, data)
        ):
            mode = AttachmentMode.TEXT
        else:
            notices.append(
                f"No puedo procesar «{name}» porque su formato no es compatible."
            )
            rejected.append(name)
            return None

        stored = StoredAttachment(
            name=name, mime_type=mime_type or "application/octet-stream",
            mode=mode, size=len(data),
        )
        await self._store.store(key, stored.name, stored.mime_type, stored.mode, data)
        return stored

    async def _next_audio_name(self, key: AttachmentKey, extension: str) -> str:
        """Genera un nombre interno neutro y único en el contexto del adjunto."""
        existing = set(await self._store.list(key))
        index = 1
        name = f"audio-adjunto-{index}{extension}"
        while name in existing:
            index += 1
            name = f"audio-adjunto-{index}{extension}"
        return name

    @staticmethod
    def _detect_image(data: bytes) -> tuple[str, str] | None:
        """Identifica formatos de imagen reconocidos por firma, no por nombre.

        Telegram `PhotoSize` no incluye MIME. Solo los formatos cuyas firmas se
        reconocen aquí pueden seguir la ruta binaria de imagen; los bytes no
        identificados nunca se etiquetan como `image/*` por conjetura.
        """
        for signature, mime_type, extension in _IMAGE_SIGNATURES:
            if data.startswith(signature):
                return mime_type, extension
        if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp", ".webp"
        return None

    def _extractable(self, name: str, data: bytes) -> bool:
        """Indica si el contenido puede enviarse como texto (Opción A)."""
        if self._extract_text is None:
            return False
        try:
            self._extract_text(name, data)
        except AttachmentError:
            return False
        return True


def detect_audio_format(
    data: bytes, *, mime_type: str, kind: str,
) -> tuple[str, str] | None:
    """Valida formatos comunes por firma; MIME y nombre no bastan solos."""
    if data.startswith(b"OggS"):
        return "audio/ogg", ".ogg"
    if data.startswith(b"fLaC"):
        return "audio/flac", ".flac"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav", ".wav"
    if len(data) >= 3 and data.startswith(b"ID3"):
        return "audio/mpeg", ".mp3"
    if (len(data) >= 4 and data[0] == 0xFF and data[1] & 0xF6 == 0xF0
            and (mime_type in _AUDIO_AAC_MIME_TYPES or kind == "audio")):
        return "audio/aac", ".aac"
    if (len(data) >= 4 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
            and ((data[1] >> 1) & 0x03) != 0):
        return "audio/mpeg", ".mp3"
    if (len(data) >= 12 and data[4:8] == b"ftyp"
            and (kind == "audio" or mime_type in _AUDIO_MP4_MIME_TYPES)):
        return "audio/mp4", ".m4a"
    if (data.startswith(b"\x1a\x45\xdf\xa3")
            and (kind == "audio" or mime_type == "audio/webm")):
        return "audio/webm", ".webm"
    return None
