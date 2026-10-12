"""Handler de Telegram: normaliza el update y delega en el caso de uso."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from collections.abc import Sequence
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from io import BytesIO
from typing import AsyncIterator

from telegram import Bot, Message, MessageEntity, Update, User
from telegram.constants import ChatAction
from telegram.error import RetryAfter, TelegramError
from telegram.ext import ContextTypes

from oraculo.application.conversation import ConversationService
from oraculo.application.pi_mode import PI_PREFIX, PiModeService, is_pi_command
from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.models import ChatKind, IncomingMessage
from oraculo.domain.pi import PiSessionState
from oraculo.telegram.attachments import download_attachment, normalize_attachments
from oraculo.telegram.media_groups import DEFAULT_WINDOW_S, MediaGroupBuffer
from oraculo.telegram.session_locks import (
    SessionUpdateLockPool,
    session_key_for_chat,
)

logger = logging.getLogger(__name__)

TYPING_START_DELAY_S = 0.5
TYPING_REFRESH_S = 4.0
PI_STREAM_EDIT_INTERVAL_S = 1.0
PI_TOOL_NOTICE_EDIT_INTERVAL_S = 2.0
TELEGRAM_MAX_CHARS = 4096


@dataclass(eq=False)
class _PendingToolNotice:
    message: Message
    notice: str
    content: str
    repetitions: int
    next_edit_at: float


def chat_kind(chat_type: str) -> ChatKind:
    """Mapea el tipo de chat de Telegram a `ChatKind`."""
    if chat_type == "private":
        return ChatKind.PRIVATE
    if chat_type in ("group", "supergroup"):
        return ChatKind.GROUP
    return ChatKind.OTHER


def mentions_bot(message: Message, bot_username: str) -> bool:
    """Indica si el mensaje menciona al bot por `@usuario` (texto o caption)."""
    if not bot_username:
        return False
    target = f"@{bot_username.lower()}"
    for entity in message.entities or ():
        if entity.type == MessageEntity.MENTION:
            if message.parse_entity(entity).lower() == target:
                return True
    for entity in message.caption_entities or ():
        if entity.type == MessageEntity.MENTION:
            if message.parse_caption_entity(entity).lower() == target:
                return True
    return False


def strip_mention(text: str, bot_username: str) -> str:
    """Elimina la mención al bot del texto antes de enviarlo a la IA."""
    if not bot_username:
        return text
    pattern = re.compile(rf"@{re.escape(bot_username)}\b", re.IGNORECASE)
    return pattern.sub("", text).strip()


def reply_context(message: Message, bot_id: int | None) -> tuple[bool, str | None]:
    """Indica si el mensaje responde a un mensaje del bot y, si es así, su texto.

    Solo se considera contexto el mensaje citado del **propio bot**: nunca el de
    terceros.
    """
    quoted = message.reply_to_message
    if quoted is None or bot_id is None:
        return False, None
    author = quoted.from_user
    if author is None or author.id != bot_id:
        return False, None
    return True, quoted.text


def display_name(user: User) -> str | None:
    """Nombre visible del usuario, para atribuir sus mensajes en grupos."""
    return user.first_name or user.username or None


async def _reply(message: Message, text: str, chat_id: int) -> bool:
    """Envía fragmentos seguros para Telegram; devuelve False ante un fallo."""
    try:
        for start in range(0, len(text), 1900):
            chunk = text[start:start + 1900]
            if message.message_thread_id is None:
                await message.reply_text(chunk)
            else:
                await message.reply_text(
                    chunk, message_thread_id=message.message_thread_id
                )
    except TelegramError:
        logger.error("No se pudo enviar la respuesta chat_id=%s", chat_id)
        return False
    return True


def _retry_after_s(exc: RetryAfter) -> float:
    """Segundos de espera de un `RetryAfter` (PTB lo expone como int o timedelta)."""
    value = exc.retry_after
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)


async def _edit_final(sent: Message, text: str, chat_id: int) -> None:
    """Aplica la edición final; ante flood control espera lo indicado y reintenta una vez."""
    try:
        await sent.edit_text(text)
        return
    except RetryAfter as exc:
        wait_s = _retry_after_s(exc)
        logger.warning(
            "Flood control en la respuesta final de Pi chat_id=%s espera=%.1fs",
            chat_id,
            wait_s,
        )
        await asyncio.sleep(wait_s)
    except TelegramError as exc:
        logger.warning(
            "No se pudo editar la respuesta final de Pi chat_id=%s (%s)",
            chat_id,
            type(exc).__name__,
        )
        return
    try:
        await sent.edit_text(text)
    except TelegramError as exc:
        logger.warning(
            "Reintento fallido de la respuesta final de Pi chat_id=%s (%s)",
            chat_id,
            type(exc).__name__,
        )


def _render_tool_notice(content: str, repetitions: int) -> str:
    """Añade repeticiones sin rebasar el límite de Telegram."""
    available = max(0, TELEGRAM_MAX_CHARS - len(content))
    visible = min(repetitions, available // 2)
    rendered = content + " …" * visible
    overflow = repetitions - visible
    if overflow:
        marker = f" … (+{overflow} repeticiones)"
        rendered = rendered[: TELEGRAM_MAX_CHARS - len(marker)] + marker
    return rendered


async def _send_tool_notice(
    message: Message, content: str, chat_id: int
) -> Message | None:
    """Envía un aviso y devuelve el mensaje para poder editarlo si se repite."""
    try:
        if message.message_thread_id is None:
            return await message.reply_text(content)
        return await message.reply_text(
            content, message_thread_id=message.message_thread_id
        )
    except TelegramError as exc:
        logger.warning(
            "No se pudo enviar aviso de herramienta de Pi chat_id=%s (%s)",
            chat_id,
            type(exc).__name__,
        )
        return None


async def _edit_tool_notice(notice: _PendingToolNotice, chat_id: int) -> None:
    """Aplica los marcadores pendientes; conserva el contador ante RetryAfter."""
    rendered = _render_tool_notice(notice.content, notice.repetitions)
    try:
        await notice.message.edit_text(rendered)
    except RetryAfter as exc:
        wait_s = _retry_after_s(exc)
        notice.next_edit_at = time.monotonic() + max(
            wait_s, PI_TOOL_NOTICE_EDIT_INTERVAL_S
        )
        logger.warning(
            "Flood control al editar aviso de Pi chat_id=%s espera=%.1fs",
            chat_id,
            wait_s,
        )
    except TelegramError as exc:
        logger.warning(
            "No se pudo editar aviso de Pi chat_id=%s (%s)",
            chat_id,
            type(exc).__name__,
        )
        notice.repetitions = 0
    else:
        notice.content = rendered
        notice.repetitions = 0
        notice.next_edit_at = time.monotonic() + PI_TOOL_NOTICE_EDIT_INTERVAL_S


async def _flush_tool_notices(
    pending: list[_PendingToolNotice], chat_id: int, *, wait: bool
) -> None:
    """Edita avisos vencidos y, al cerrar el turno, espera por los restantes."""
    while pending:
        now = time.monotonic()
        pending[:] = [notice for notice in pending if notice.repetitions > 0]
        if not pending:
            return
        due = [notice for notice in pending if now >= notice.next_edit_at]
        if not due:
            if not wait:
                return
            next_edit_at = min(notice.next_edit_at for notice in pending)
            await asyncio.sleep(max(0.0, next_edit_at - now))
            continue
        for notice in due:
            await _edit_tool_notice(notice, chat_id)


@asynccontextmanager
async def typing_indicator(
    bot: Bot,
    chat_id: int,
    *,
    start_delay_s: float = TYPING_START_DELAY_S,
    refresh_s: float = TYPING_REFRESH_S,
) -> AsyncIterator[None]:
    """Muestra «escribiendo…» en el chat mientras dura el bloque `async with`.

    La primera acción sale tras `start_delay_s` (demora de gracia que evita el
    parpadeo en turnos resueltos localmente) y se renueva cada `refresh_s`,
    porque Telegram caduca el estado a los ~5 s. Un fallo de envío detiene el
    bucle sin interrumpir el procesamiento; al salir siempre se cancela la tarea.
    """
    async def _pump() -> None:
        try:
            await asyncio.sleep(start_delay_s)
            while True:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
                await asyncio.sleep(refresh_s)
        except TelegramError:
            logger.warning("No se pudo enviar la acción de chat chat_id=%s", chat_id)

    task = asyncio.create_task(_pump())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def _download_items(
    bot: Bot, found: Sequence[IncomingAttachment]
) -> tuple[tuple[IncomingAttachment, bytes | None], ...]:
    """Descarga cada adjunto; `None` marca los que fallaron o exceden el peso."""
    items: list[tuple[IncomingAttachment, bytes | None]] = []
    for item in found:
        items.append((item, await download_attachment(bot, item)))
    return tuple(items)


def _is_pi_image(attachment: IncomingAttachment) -> bool:
    """Reconoce fotos y documentos identificados como imágenes para Pi."""
    mime_type = (attachment.mime_type or "").split(";", 1)[0].strip().lower()
    file_name = (attachment.file_name or "").lower()
    supported_suffix = file_name.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
    return (
        attachment.kind == "photo"
        or mime_type.startswith("image/")
        or supported_suffix
    )


async def _download_pi_images(
    bot: Bot, found: Sequence[IncomingAttachment]
) -> tuple[tuple[bytes, ...], int]:
    """Descarga solo imágenes para Pi y cuenta las que no pudieron obtenerse."""
    images: list[bytes] = []
    failed = 0
    for attachment in found:
        data = await download_attachment(bot, attachment)
        if data is None:
            failed += 1
        else:
            images.append(data)
    return tuple(images), failed


class OraculoMessageHandler:
    """Procesa cada mensaje entrante y responde si el caso de uso lo indica."""

    def __init__(
        self,
        service: ConversationService,
        *,
        pi_mode_service: PiModeService | None = None,
        typing_start_delay_s: float = TYPING_START_DELAY_S,
        typing_refresh_s: float = TYPING_REFRESH_S,
        media_window_s: float = DEFAULT_WINDOW_S,
        pi_stream_edit_interval_s: float = PI_STREAM_EDIT_INTERVAL_S,
        session_update_locks: SessionUpdateLockPool | None = None,
    ) -> None:
        self._service = service
        self._pi_mode_service = pi_mode_service
        self._session_update_locks = (
            session_update_locks or SessionUpdateLockPool()
        )
        self._pi_stream_edit_interval_s = pi_stream_edit_interval_s
        self._typing_start_delay_s = typing_start_delay_s
        self._typing_refresh_s = typing_refresh_s
        self._media_window_s = media_window_s
        self._media_buffer = MediaGroupBuffer(window_s=media_window_s)
        self._album_tasks: dict[tuple[int, str], asyncio.Task[None]] = {}
        self._album_incoming: dict[tuple[int, str], tuple[IncomingMessage, Message]] = {}

    async def __call__(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if message is None or chat is None or user is None:
            return
        if user.is_bot:
            logger.debug("Mensaje de otro bot ignorado user_id=%s", user.id)
            return

        bot_username = context.bot.username or ""
        raw_text = getattr(message, "text", None)
        if raw_text is None:
            raw_text = getattr(message, "caption", None)
        replies_to_bot, quoted_text = reply_context(message, context.bot.id)
        incoming = IncomingMessage(
            chat_id=chat.id,
            user_id=user.id,
            chat_type=chat_kind(chat.type),
            text=strip_mention(raw_text, bot_username) if raw_text is not None else None,
            mentions_bot=mentions_bot(message, bot_username),
            replies_to_bot=replies_to_bot,
            quoted_text=quoted_text,
            user_display_name=display_name(user),
        )

        session_key = session_key_for_chat(
            incoming.chat_type, incoming.chat_id, incoming.user_id
        )
        if session_key is None:
            await self._handle_incoming(incoming, message, context)
            return
        async with self._session_update_locks.hold(session_key):
            await self._handle_incoming(incoming, message, context)

    async def _handle_incoming(
        self,
        incoming: IncomingMessage,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Procesa un update mientras el lock de su conversación está tomado."""
        chat_id = incoming.chat_id

        if (
            self._pi_mode_service is None
            and incoming.text in ("/agent", "/exitagent")
        ):
            await _reply(message, "El modo agente no está habilitado.", chat_id)
            return

        if self._pi_mode_service is not None:
            text = incoming.text or ""
            pi_text = text in ("/agent", "/exitagent", "/stoppi") or is_pi_command(text)
            # La sesión Pi es privada: en grupos nunca captura mensajes del autor.
            pi_session = (
                incoming.chat_type is ChatKind.PRIVATE
                and self._pi_mode_service.is_in_pi_mode(incoming.user_id)
            )
            if pi_text or pi_session:
                access = self._service.pi_access(incoming)
                if access.allowed:
                    pi_attachments = normalize_attachments(message) if pi_session else ()
                    voice_attachment = next(
                        (item for item in pi_attachments if item.kind == "voice"),
                        None,
                    )
                    state = self._pi_mode_service.get_state(incoming.user_id)
                    if voice_attachment is not None:
                        if state not in (PiSessionState.PI_ACTIVO, PiSessionState.PI_ERROR):
                            return
                        voice_data = await download_attachment(context.bot, voice_attachment)
                        await self._handle_pi_mode(
                            incoming,
                            message,
                            voice=(voice_attachment, voice_data),
                        )
                        return

                    images: list[bytes] | None = None
                    image_attachments: tuple[IncomingAttachment, ...] = ()
                    image_failed = 0
                    active_pi_turn = (
                        pi_session
                        and state in (PiSessionState.PI_ACTIVO, PiSessionState.PI_ERROR)
                        and text not in {"/agent", "/exitagent", "/stoppi", "/new", "/reset"}
                        and not is_pi_command(text)
                    )
                    if active_pi_turn:
                        image_attachments = tuple(
                            item for item in pi_attachments if _is_pi_image(item)
                        )
                        if image_attachments:
                            downloaded, image_failed = await _download_pi_images(
                                context.bot, image_attachments
                            )
                            images = list(downloaded)
                    if image_failed:
                        await _reply(
                            message,
                            f"{PI_PREFIX} No pude descargar una o más imágenes; las omití.",
                            chat_id,
                        )
                    if active_pi_turn and image_attachments and not images:
                        return
                    await self._handle_pi_mode(incoming, message, images=images)
                elif access.reply is not None:
                    await _reply(message, access.reply, chat_id)
                return

        if incoming.text in ("/new", "/reset"):
            reply = await self._service.handle_reset(incoming)
        else:
            found = normalize_attachments(message)
            if found:
                await self._dispatch_attachments(incoming, message, context.bot, found)
                return
            attachment = await self._service.handle_attachment(incoming)
            if attachment is not None:
                filename, content = attachment
                try:
                    await message.reply_document(document=BytesIO(content), filename=filename)
                except TelegramError:
                    logger.error("No se pudo enviar el documento chat_id=%s", chat_id)
                    await _reply(message, "No se pudo confirmar la entrega del archivo. Inténtalo de nuevo.", chat_id)
                    return
                notice = f"Archivo entregado: {filename}"
                if incoming.chat_type is ChatKind.GROUP:
                    notice += " (visible para todos en este grupo)"
                await _reply(message, notice + ".", chat_id)
                return
            async with typing_indicator(
                context.bot,
                chat_id,
                start_delay_s=self._typing_start_delay_s,
                refresh_s=self._typing_refresh_s,
            ):
                reply = await self._service.handle(incoming)
        if reply is None:
            return
        sent = await _reply(message, reply, chat_id)
        self._service.preview_delivery(incoming, reply, sent)

    async def _handle_pi_mode(
        self,
        incoming: IncomingMessage,
        message: Message,
        *,
        images: list[bytes] | None = None,
        voice: tuple[IncomingAttachment, bytes | None] | None = None,
    ) -> None:
        """Procesa texto, voz o imágenes en modo Pi y renderiza sus eventos."""
        if self._pi_mode_service is None:
            return
        text = (incoming.text or "").strip()
        if voice is None and not text:
            if not images:
                return
            text = "Analiza las imágenes adjuntas."

        chat_id = incoming.chat_id
        stream_message: Message | None = None
        stream_buffer = ""
        # Texto visible en Telegram y momento a partir del cual se puede volver a editar.
        shown = ""
        next_edit_at = 0.0
        pending_tool_notices: list[_PendingToolNotice] = []
        last_tool_notice: _PendingToolNotice | None = None

        def close_tool_notice_group() -> None:
            nonlocal last_tool_notice
            if last_tool_notice is not None and last_tool_notice.repetitions > 0:
                if not any(item is last_tool_notice for item in pending_tool_notices):
                    pending_tool_notices.append(last_tool_notice)
            last_tool_notice = None

        if voice is not None:
            events = self._pi_mode_service.handle_voice(
                incoming.user_id,
                voice[0],
                voice[1],
                caption=incoming.text,
            )
        else:
            events = self._pi_mode_service.handle(
                incoming.user_id, text, images=images
            )

        async for event in events:
            await _flush_tool_notices(pending_tool_notices, chat_id, wait=False)
            if event.type == "tool_notice":
                if (
                    last_tool_notice is not None
                    and last_tool_notice.notice != event.content
                ):
                    close_tool_notice_group()
                if (
                    last_tool_notice is not None
                    and last_tool_notice.notice == event.content
                ):
                    last_tool_notice.repetitions += 1
                    if not any(
                        item is last_tool_notice for item in pending_tool_notices
                    ):
                        pending_tool_notices.append(last_tool_notice)
                    await _flush_tool_notices(pending_tool_notices, chat_id, wait=False)
                else:
                    sent_notice = await _send_tool_notice(
                        message, event.content, chat_id
                    )
                    if sent_notice is None:
                        last_tool_notice = None
                    else:
                        last_tool_notice = _PendingToolNotice(
                            message=sent_notice,
                            notice=event.content,
                            content=event.content,
                            repetitions=0,
                            next_edit_at=(
                                time.monotonic() + PI_TOOL_NOTICE_EDIT_INTERVAL_S
                            ),
                        )
                continue

            close_tool_notice_group()
            if event.type in {"response", "text_final", "error"}:
                await _flush_tool_notices(pending_tool_notices, chat_id, wait=True)

            if event.type == "response":
                await _reply(message, event.content, chat_id)
            elif event.type == "text_delta":
                stream_buffer += event.content
                rendered = f"{PI_PREFIX} {stream_buffer}"[:4096]
                now = time.monotonic()
                if stream_message is None:
                    stream_message = await message.reply_text(rendered)
                    shown = rendered
                    next_edit_at = now + self._pi_stream_edit_interval_s
                elif now >= next_edit_at and rendered != shown:
                    # Agrupa fragmentos: Telegram limita las ediciones por chat.
                    try:
                        await stream_message.edit_text(rendered)
                        shown = rendered
                        next_edit_at = now + self._pi_stream_edit_interval_s
                    except RetryAfter as exc:
                        wait_s = _retry_after_s(exc)
                        logger.warning(
                            "Flood control al editar streaming de Pi chat_id=%s espera=%.1fs",
                            chat_id,
                            wait_s,
                        )
                        next_edit_at = now + wait_s
                    except TelegramError as exc:
                        logger.warning(
                            "No se pudo editar el streaming de Pi chat_id=%s (%s)",
                            chat_id,
                            type(exc).__name__,
                        )
                        next_edit_at = now + self._pi_stream_edit_interval_s
            elif event.type == "text_final":
                final = f"{PI_PREFIX} {event.content}"[:4096]
                if stream_message is None:
                    await _reply(message, final, chat_id)
                elif final != shown:
                    await _edit_final(stream_message, final, chat_id)
                stream_message = None
                stream_buffer = ""
                shown = ""
            elif event.type == "error":
                await _reply(message, event.content, incoming.chat_id)

        close_tool_notice_group()
        await _flush_tool_notices(pending_tool_notices, chat_id, wait=True)

    async def _dispatch_attachments(
        self,
        incoming: IncomingMessage,
        message: Message,
        bot: Bot,
        found: tuple[IncomingAttachment, ...],
    ) -> None:
        """Reparte los adjuntos entre la interacción inmediata y los álbumes."""
        media_group_id = getattr(message, "media_group_id", None)
        if media_group_id is None:
            items = await _download_items(bot, found)
            await self._process_attachment_items(incoming, message, bot, items)
            return
        key = (incoming.chat_id, media_group_id)
        self._album_incoming.setdefault(key, (incoming, message))
        for item in found:
            self._media_buffer.add(
                chat_id=incoming.chat_id,
                media_group_id=media_group_id,
                attachment=item,
            )
        if key not in self._album_tasks:
            self._album_tasks[key] = asyncio.create_task(
                self._flush_album(key, bot)
            )

    async def _flush_album(self, key: tuple[int, str], bot: Bot) -> None:
        """Procesa un álbum cuando se cierra su ventana de agrupación."""
        task = asyncio.current_task()
        try:
            await asyncio.sleep(self._media_window_s)
            meta = self._album_incoming.get(key)
            if meta is None:
                return
            incoming, message = meta
            session_key = session_key_for_chat(
                incoming.chat_type, incoming.chat_id, incoming.user_id
            )
            if session_key is None:
                return
            async with self._session_update_locks.hold(session_key):
                chat_id, media_group_id = key
                album = self._media_buffer.flush_group(
                    chat_id=chat_id, media_group_id=media_group_id
                )
                if album is None:
                    if self._media_buffer.contains_group(
                        chat_id=chat_id, media_group_id=media_group_id
                    ):
                        self._album_tasks[key] = asyncio.create_task(
                            self._flush_album(key, bot)
                        )
                    else:
                        self._album_incoming.pop(key, None)
                    return
                self._album_tasks.pop(key, None)
                self._album_incoming.pop(key, None)
                _, _, items = album
                downloaded = await _download_items(bot, items)
                await self._process_attachment_items(
                    incoming, message, bot, downloaded
                )
        finally:
            if self._album_tasks.get(key) is task:
                self._album_tasks.pop(key, None)

    async def _process_attachment_items(
        self,
        incoming: IncomingMessage,
        message: Message,
        bot: Bot,
        items: tuple[tuple[IncomingAttachment, bytes | None], ...],
    ) -> None:
        async with typing_indicator(
            bot,
            incoming.chat_id,
            start_delay_s=self._typing_start_delay_s,
            refresh_s=self._typing_refresh_s,
        ):
            reply = await self._service.handle_attachments(incoming, items)
        if reply is None:
            return
        sent = await _reply(message, reply, incoming.chat_id)
        self._service.preview_delivery(incoming, reply, sent)


class OraculoHelpHandler:
    """Muestra la ayuda estática de los comandos disponibles."""

    _HELP_TEXT = (
        "Comandos disponibles:\n"
        "/new — Inicia una conversación nueva.\n"
        "/agent — Activa el modo agente y permite elegir un proyecto (solo en privado).\n"
        "/exitagent — Sale del modo agente.\n"
        "/help — Muestra esta ayuda."
    )

    async def __call__(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        del context
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if message is None or chat is None or user is None or user.is_bot:
            return
        await _reply(message, self._HELP_TEXT, chat.id)


class OraculoResetHandler:
    """Atiende los comandos de reinicio de sesión (`/new` y `/reset`)."""

    def __init__(self, service: ConversationService) -> None:
        self._service = service

    async def __call__(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        if message is None or chat is None or user is None:
            return
        if user.is_bot:
            return
        incoming = IncomingMessage(
            chat_id=chat.id,
            user_id=user.id,
            chat_type=chat_kind(chat.type),
            text=None,
            mentions_bot=False,
            replies_to_bot=False,
            quoted_text=None,
            user_display_name=display_name(user),
        )
        reply = await self._service.handle_reset(incoming)
        if reply is None:
            return
        await _reply(message, reply, chat.id)


class OraculoCommandHandler:
    """Enruta `/help` y delega los comandos de sesión al handler de reinicio."""

    def __init__(
        self,
        service: ConversationService,
        *,
        session_update_locks: SessionUpdateLockPool | None = None,
    ) -> None:
        self._session_update_locks = (
            session_update_locks or SessionUpdateLockPool()
        )
        self._help_handler = OraculoHelpHandler()
        self._reset_handler = OraculoResetHandler(service)

    async def __call__(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        chat = update.effective_chat
        user = update.effective_user
        if chat is None or user is None:
            await self._dispatch(update, context)
            return
        key = session_key_for_chat(chat_kind(chat.type), chat.id, user.id)
        if key is None:
            await self._dispatch(update, context)
            return
        async with self._session_update_locks.hold(key):
            await self._dispatch(update, context)

    async def _dispatch(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        message = update.effective_message
        text = message.text if message is not None else None
        command = (
            text.split(maxsplit=1)[0].split("@", 1)[0].lower() if text else ""
        )
        if command == "/help":
            await self._help_handler(update, context)
            return
        await self._reset_handler(update, context)
