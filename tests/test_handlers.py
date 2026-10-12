"""Tests del adaptador de Telegram (Req 1.5, 3.2, 4.4).

Verifica el mapeo update -> `IncomingMessage`, la detección/recorte de la mención
y la tolerancia a fallos de envío.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from io import BytesIO
from types import SimpleNamespace
from typing import Optional

import pytest
from telegram import Bot, Chat, Message, MessageEntity
from telegram.error import TelegramError

from oraculo.domain.models import ChatKind, IncomingMessage
from oraculo.domain.pi import PiAccess, PiEvent, PiSessionState
from oraculo.telegram.handlers import (
    OraculoCommandHandler,
    OraculoMessageHandler,
    OraculoResetHandler,
    chat_kind,
    mentions_bot,
    reply_context,
    strip_mention,
    _reply,
)
from oraculo.telegram.session_locks import SessionUpdateLockPool

BOT_USERNAME = "oraculo_bot"
BOT_ID = 424242


class FakeReplyMessage:
    """Mensaje falso con el mínimo que usa el handler."""

    def __init__(
        self,
        text: str,
        entities: tuple = (),
        reply_to_message: object = None,
        *,
        caption: str | None = None,
        caption_entities: tuple = (),
    ) -> None:
        self.text = text
        self.entities = entities
        self.caption = caption
        self.caption_entities = caption_entities
        self.reply_to_message = reply_to_message
        self.message_thread_id: int | None = None
        self.replies: list[str] = []
        self.documents: list[tuple[str, bytes]] = []
        self._telegram_message = Message(
            message_id=1,
            date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            chat=Chat(id=100, type="private"),
            text=text,
            entities=entities,
            caption=caption,
            caption_entities=caption_entities,
        )

    def parse_entity(self, entity: MessageEntity) -> str:
        return self._telegram_message.parse_entity(entity)

    def parse_caption_entity(self, entity: MessageEntity) -> str:
        return self._telegram_message.parse_caption_entity(entity)

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)

    async def reply_document(self, document: BytesIO, *, filename: str) -> None:
        self.documents.append((filename, document.read()))


class FailingReplyMessage(FakeReplyMessage):
    async def reply_text(self, text: str) -> None:
        raise TelegramError("fallo de envío")


async def test_telegram_error_does_not_log_document_or_code(caplog):
    class SensitiveFailure(FakeReplyMessage):
        async def reply_text(self, text: str) -> None:
            raise TelegramError("DOCUMENT_SECRET confirma SECRET_CODE")

    assert await _reply(SensitiveFailure("hola"), "respuesta", 1) is False
    assert "DOCUMENT_SECRET" not in caplog.text
    assert "SECRET_CODE" not in caplog.text


class RecordingService:
    def __init__(
        self, reply: Optional[str] = "ok", reset_reply: Optional[str] = "sesión nueva"
    ) -> None:
        self.reply = reply
        self.reset_reply = reset_reply
        self.received: list[IncomingMessage] = []
        self.reset_received: list[IncomingMessage] = []
        self.attachment_reply: tuple[str, bytes] | None = None
        self.attachment_received: list[IncomingMessage] = []

    def preview_delivery(self, message: IncomingMessage, reply: str, sent: bool) -> None:
        pass

    async def handle_attachment(self, message: IncomingMessage) -> tuple[str, bytes] | None:
        self.attachment_received.append(message)
        return self.attachment_reply

    async def handle(self, message: IncomingMessage) -> Optional[str]:
        self.received.append(message)
        return self.reply

    async def handle_reset(self, message: IncomingMessage) -> Optional[str]:
        self.reset_received.append(message)
        return self.reset_reply


class BlockingRecordingService(RecordingService):
    """Permite controlar el intercalado de las llamadas a conversación."""

    def __init__(self) -> None:
        super().__init__(reply=None)
        self.entered: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        self.release = asyncio.Event()
        self.active = 0
        self.max_active = 0
        self.reset_entered = asyncio.Event()
        self.reset_release = asyncio.Event()

    async def handle(self, message: IncomingMessage) -> Optional[str]:
        self.received.append(message)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.entered.put_nowait(message)
        try:
            await self.release.wait()
        finally:
            self.active -= 1
        return None

    def pi_access(self, message: IncomingMessage) -> PiAccess:
        del message
        return PiAccess(allowed=True, reply=None)

    async def handle_reset(self, message: IncomingMessage) -> Optional[str]:
        self.reset_received.append(message)
        self.reset_entered.set()
        await self.reset_release.wait()
        return None


class BlockingPiModeService:
    def __init__(self) -> None:
        self.entered: asyncio.Queue[str] = asyncio.Queue()
        self.release = asyncio.Event()

    def is_in_pi_mode(self, user_id: int) -> bool:
        del user_id
        return True

    def get_state(self, user_id: int) -> PiSessionState:
        del user_id
        return PiSessionState.PI_ACTIVO

    async def handle(
        self, user_id: int, text: str, *, images: list[bytes] | None = None
    ) -> AsyncIterator[PiEvent]:
        del user_id, images
        self.entered.put_nowait(text)
        await self.release.wait()
        yield PiEvent(type="settled", content="")


class BlockingWorkspaceService(RecordingService):
    def __init__(self) -> None:
        super().__init__(reply="respuesta posterior")
        self.workspace_entered = asyncio.Event()
        self.workspace_release = asyncio.Event()

    async def handle_attachment(
        self, message: IncomingMessage
    ) -> tuple[str, bytes] | None:
        if message.text != "envíame nota.md":
            return None
        self.attachment_received.append(message)
        self.workspace_entered.set()
        await self.workspace_release.wait()
        return "nota.md", b"contenido"


class FakeBot:
    """Bot falso que registra las acciones de chat enviadas."""

    def __init__(
        self, username: str, bot_id: int = BOT_ID, *, fail_send_action: bool = False
    ) -> None:
        self.username = username
        self.id = bot_id
        self.actions: list[tuple[int, str]] = []
        self._fail_send_action = fail_send_action

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))
        if self._fail_send_action:
            raise TelegramError("fallo de chat action")


class FakeContext:
    def __init__(
        self, username: str, bot_id: int = BOT_ID, *, fail_send_action: bool = False
    ) -> None:
        self.bot = FakeBot(username, bot_id, fail_send_action=fail_send_action)


def make_message(
    text: str,
    mention_username: Optional[str] = None,
    reply_to_message: object = None,
) -> FakeReplyMessage:
    entities: tuple[MessageEntity, ...] = ()
    if mention_username is not None:
        token = f"@{mention_username}"
        offset = len(text[:text.index(token)].encode("utf-16-le")) // 2
        entities = (
            MessageEntity(type=MessageEntity.MENTION, offset=offset, length=len(token)),
        )
    return FakeReplyMessage(text, entities, reply_to_message)


def make_update(
    message: object,
    chat_type: str = "private",
    chat_id: int = 100,
    user_id: int = 1,
    is_bot: bool = False,
    first_name: str = "Ana",
    username: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_user=SimpleNamespace(
            id=user_id, is_bot=is_bot, first_name=first_name, username=username
        ),
    )


def test_mentions_bot_detecta_al_bot():
    message = make_message("hola @oraculo_bot dime", BOT_USERNAME)

    assert mentions_bot(message, BOT_USERNAME) is True  # type: ignore[arg-type]


def test_mentions_bot_no_confunde_a_otro_usuario():
    message = make_message("hola @otro dime", "otro")

    assert mentions_bot(message, BOT_USERNAME) is False  # type: ignore[arg-type]


@pytest.mark.parametrize("source", ["text", "caption"])
def test_mentions_bot_respeta_offsets_utf16_despues_de_emoji(source: str) -> None:
    text = "😀 @oraculo_bot consulta"
    entity = MessageEntity(type=MessageEntity.MENTION, offset=3, length=12)
    message = Message(
        message_id=1,
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        chat=Chat(id=-100, type="supergroup"),
        text=text if source == "text" else None,
        entities=(entity,) if source == "text" else (),
        caption=text if source == "caption" else None,
        caption_entities=(entity,) if source == "caption" else (),
    )

    assert mentions_bot(message, BOT_USERNAME) is True  # type: ignore[arg-type]


async def test_reply_preserva_message_thread_id_en_versiones_ptb_anteriores_a_21_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_messages: list[dict[str, object]] = []

    async def capture_send_message(_bot: Bot, **kwargs: object) -> None:
        sent_messages.append(kwargs)

    monkeypatch.setattr(Bot, "send_message", capture_send_message)
    bot = Bot("123:abc")
    message = Message(
        message_id=7,
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        chat=Chat(id=-100, type="supergroup"),
        text="@oraculo_bot consulta",
        message_thread_id=9876,
        is_topic_message=True,
    )
    message.set_bot(bot)

    assert await _reply(message, "respuesta", -100) is True

    assert sent_messages[0]["message_thread_id"] == 9876


def test_mentions_bot_sin_username_no_detecta():
    message = make_message("hola @oraculo_bot", BOT_USERNAME)

    assert mentions_bot(message, "") is False  # type: ignore[arg-type]


def test_strip_mention_elimina_la_mencion():
    assert strip_mention("@Oraculo_Bot dime algo", BOT_USERNAME) == "dime algo"


def test_strip_mention_no_toca_otras_menciones():
    assert strip_mention("hola @otro dime", BOT_USERNAME) == "hola @otro dime"


@pytest.mark.parametrize(
    "chat_type,expected",
    [
        ("private", ChatKind.PRIVATE),
        ("group", ChatKind.GROUP),
        ("supergroup", ChatKind.GROUP),
        ("channel", ChatKind.OTHER),
    ],
)
def test_chat_kind(chat_type, expected):
    assert chat_kind(chat_type) is expected


async def test_handler_normaliza_y_responde():
    service = RecordingService(reply="respuesta")
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("hola")

    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert len(service.received) == 1
    incoming = service.received[0]
    assert incoming.chat_type is ChatKind.PRIVATE
    assert incoming.user_id == 1
    assert incoming.chat_id == 100
    assert incoming.text == "hola"
    assert message.replies == ["respuesta"]


async def test_privado_y_grupo_del_mismo_usuario_pueden_solaparse() -> None:
    service = BlockingRecordingService()
    handler = OraculoMessageHandler(
        service, session_update_locks=SessionUpdateLockPool(), typing_start_delay_s=30
    )  # type: ignore[arg-type]
    context = FakeContext(BOT_USERNAME)
    private = asyncio.create_task(
        handler(
            make_update(make_message("privado"), user_id=1),  # type: ignore[arg-type]
            context,  # type: ignore[arg-type]
        )
    )
    private_incoming = await asyncio.wait_for(service.entered.get(), timeout=1)
    group_message = make_message("@oraculo_bot grupo", BOT_USERNAME)
    group = asyncio.create_task(
        handler(
            make_update(group_message, chat_type="group", chat_id=-100, user_id=1),
            context,  # type: ignore[arg-type]
        )
    )

    try:
        group_incoming = await asyncio.wait_for(service.entered.get(), timeout=1)
        assert private_incoming.text == "privado"
        assert group_incoming.text == "grupo"
        assert service.max_active == 2
    finally:
        service.release.set()
        await asyncio.gather(private, group)


async def test_miembros_del_mismo_grupo_se_serializan_en_orden() -> None:
    service = BlockingRecordingService()
    handler = OraculoMessageHandler(
        service, session_update_locks=SessionUpdateLockPool(), typing_start_delay_s=30
    )  # type: ignore[arg-type]
    context = FakeContext(BOT_USERNAME)
    first_message = make_message("@oraculo_bot primero", BOT_USERNAME)
    first = asyncio.create_task(
        handler(
            make_update(first_message, chat_type="supergroup", chat_id=-100, user_id=1),
            context,  # type: ignore[arg-type]
        )
    )
    first_incoming = await asyncio.wait_for(service.entered.get(), timeout=1)
    second_message = make_message("@oraculo_bot segundo", BOT_USERNAME)
    second = asyncio.create_task(
        handler(
            make_update(second_message, chat_type="supergroup", chat_id=-100, user_id=2),
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.sleep(0)

    assert first_incoming.text == "primero"
    assert service.entered.empty()
    assert service.max_active == 1

    service.release.set()
    second_incoming = await asyncio.wait_for(service.entered.get(), timeout=1)
    await asyncio.gather(first, second)

    assert second_incoming.text == "segundo"
    assert [message.text for message in service.received] == ["primero", "segundo"]


async def test_comando_reset_comparte_lock_con_mensaje_de_la_misma_sesion() -> None:
    service = BlockingRecordingService()
    locks = SessionUpdateLockPool()
    message_handler = OraculoMessageHandler(
        service, session_update_locks=locks, typing_start_delay_s=30
    )  # type: ignore[arg-type]
    command_handler = OraculoCommandHandler(
        service, session_update_locks=locks
    )  # type: ignore[arg-type]
    context = FakeContext(BOT_USERNAME)
    message_task = asyncio.create_task(
        message_handler(
            make_update(
                make_message("@oraculo_bot consulta", BOT_USERNAME),
                chat_type="group", chat_id=-100, user_id=1,
            ),  # type: ignore[arg-type]
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(service.entered.get(), timeout=1)
    reset_task = asyncio.create_task(
        command_handler(
            make_update(
                FakeReplyMessage("/new"),
                chat_type="group", chat_id=-100, user_id=2,
            ),  # type: ignore[arg-type]
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.sleep(0)

    assert not service.reset_entered.is_set()

    service.release.set()
    await asyncio.wait_for(service.reset_entered.wait(), timeout=1)
    service.reset_release.set()
    await asyncio.gather(message_task, reset_task)
    assert len(service.reset_received) == 1


async def test_modo_pi_serializa_updates_del_mismo_usuario() -> None:
    service = BlockingRecordingService()
    pi_service = BlockingPiModeService()
    handler = OraculoMessageHandler(
        service,
        pi_mode_service=pi_service,  # type: ignore[arg-type]
        session_update_locks=SessionUpdateLockPool(),
        typing_start_delay_s=30,
    )  # type: ignore[arg-type]
    context = FakeContext(BOT_USERNAME)
    first = asyncio.create_task(
        handler(
            make_update(make_message("primero"), user_id=1),  # type: ignore[arg-type]
            context,  # type: ignore[arg-type]
        )
    )
    assert await asyncio.wait_for(pi_service.entered.get(), timeout=1) == "primero"
    second = asyncio.create_task(
        handler(
            make_update(make_message("segundo"), user_id=1),  # type: ignore[arg-type]
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.sleep(0)

    assert pi_service.entered.empty()

    pi_service.release.set()
    assert await asyncio.wait_for(pi_service.entered.get(), timeout=1) == "segundo"
    await asyncio.gather(first, second)


async def test_entrega_workspace_bloquea_temporalmente_otra_operacion_del_grupo() -> None:
    service = BlockingWorkspaceService()
    handler = OraculoMessageHandler(
        service, session_update_locks=SessionUpdateLockPool(), typing_start_delay_s=30
    )  # type: ignore[arg-type]
    context = FakeContext(BOT_USERNAME)
    delivery_message = make_message("@oraculo_bot envíame nota.md", BOT_USERNAME)
    delivery = asyncio.create_task(
        handler(
            make_update(
                delivery_message, chat_type="group", chat_id=-100, user_id=1
            ),  # type: ignore[arg-type]
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.wait_for(service.workspace_entered.wait(), timeout=1)
    other_message = make_message("@oraculo_bot consulta", BOT_USERNAME)
    other = asyncio.create_task(
        handler(
            make_update(other_message, chat_type="group", chat_id=-100, user_id=2),
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.sleep(0)

    assert not service.received

    service.workspace_release.set()
    await asyncio.gather(delivery, other)

    assert delivery_message.documents == [("nota.md", b"contenido")]
    assert len(service.received) == 1
    assert service.received[0].text == "consulta"


@pytest.mark.parametrize(
    "chat_type,chat_id,text,mention",
    [
        ("private", 100, "consulta", None),
        ("group", -100, "@oraculo_bot consulta", BOT_USERNAME),
    ],
)
async def test_handler_entrega_el_error_en_el_chat_de_origen(
    chat_type: str, chat_id: int, text: str, mention: str | None,
) -> None:
    reply = "OOps: cuota agotada"
    service = RecordingService(reply=reply)
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = make_message(text, mention)

    await handler(make_update(message, chat_type=chat_type, chat_id=chat_id),
                  FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert message.replies == [reply]
    assert len(service.received) == 1
    assert service.received[0].chat_id == chat_id


async def test_handler_envia_respuesta_con_aviso_de_compactacion():
    reply = "respuesta IA\n\nHe resumido el contexto de esta sesión para poder continuar."
    handler = OraculoMessageHandler(RecordingService(reply=reply))  # type: ignore[arg-type]
    message = FakeReplyMessage("consulta")
    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]
    assert message.replies == [reply]


async def test_comando_pi_informa_si_la_funcion_no_esta_habilitada():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("/agent")

    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert message.replies == ["El modo agente no está habilitado."]
    assert service.received == []


async def test_handler_ignora_cuando_el_servicio_no_responde():
    service = RecordingService(reply=None)
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("hola")

    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert message.replies == []


async def test_handler_envia_adjunto_validado_en_grupo_sin_consultar_ia():
    service = RecordingService()
    service.attachment_reply = ("archivo.md", b"contenido")
    message = make_message("@oraculo_bot envíame archivo.md", BOT_USERNAME)
    await OraculoMessageHandler(service)(
        make_update(message, chat_type="group", chat_id=-100), FakeContext(BOT_USERNAME)
    )  # type: ignore[arg-type]
    assert service.attachment_received[0].text == "envíame archivo.md"
    assert service.received == []
    assert message.documents == [("archivo.md", b"contenido")]
    assert message.replies == ["Archivo entregado: archivo.md (visible para todos en este grupo)."]


async def test_handler_fallo_de_adjunto_no_confirma_entrega():
    class FailingDocument(FakeReplyMessage):
        async def reply_document(self, document: BytesIO, *, filename: str) -> None:
            raise TelegramError("envío fallido")

    service = RecordingService()
    service.attachment_reply = ("archivo.md", b"contenido")
    message = FailingDocument("envíame archivo.md")
    await OraculoMessageHandler(service)(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]
    assert message.replies == ["No se pudo confirmar la entrega del archivo. Inténtalo de nuevo."]


async def test_handler_ambiguous_attachment_asks_without_sending_document():
    service = RecordingService(reply="Indica un único archivo .md o .txt.")
    message = FakeReplyMessage("envíame")
    await OraculoMessageHandler(service)(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]
    assert message.documents == []
    assert message.replies == ["Indica un único archivo .md o .txt."]


async def test_handler_recorta_la_mencion_en_grupo():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = make_message("@oraculo_bot dime algo", BOT_USERNAME)

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100),
        FakeContext(BOT_USERNAME),
    )

    incoming = service.received[0]
    assert incoming.chat_type is ChatKind.GROUP
    assert incoming.mentions_bot is True
    assert incoming.text == "dime algo"


@pytest.mark.parametrize("command", ["/new", "/reset"])
async def test_handler_reinicia_con_mencion_antes_del_comando(command):
    service = RecordingService(reset_reply="sesión nueva")
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = make_message(f"@Oraculo_Bot {command}", "Oraculo_Bot")

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100),
        FakeContext(BOT_USERNAME),
    )

    assert len(service.reset_received) == 1
    assert service.reset_received[0].text == command
    assert service.received == []
    assert message.replies == ["sesión nueva"]


async def test_handler_no_reinicia_si_el_comando_no_es_el_unico_texto():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = make_message("@oraculo_bot hola /new", BOT_USERNAME)

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100),
        FakeContext(BOT_USERNAME),
    )

    assert service.reset_received == []
    assert service.received[0].text == "hola /new"
    assert message.replies == ["ok"]


async def test_handler_registra_fallo_de_envio_sin_propagar():
    service = RecordingService(reply="respuesta")
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FailingReplyMessage("hola")

    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert message.replies == []


async def test_reply_splits_long_unicode_preview_and_reports_success():
    message = FakeReplyMessage("consulta")
    text = "😀" * 3200 + " confirma TOKEN"
    assert await _reply(message, text, 100) is True  # type: ignore[arg-type]
    assert "".join(message.replies) == text
    assert len(message.replies) > 1
    assert all(len(chunk.encode("utf-16-le")) // 2 < 4096 for chunk in message.replies)


async def test_failed_preview_chunk_revokes_proposal_before_next_confirm():
    class FailSecond(FakeReplyMessage):
        async def reply_text(self, text: str) -> None:
            if len(self.replies) == 1:
                raise TelegramError("fallo parcial")
            await super().reply_text(text)

    class PreviewService(RecordingService):
        def __init__(self):
            super().__init__(reply="vista previa " + "😀" * 3200 + " confirma TOKEN")
            self.deliveries = []

        def preview_delivery(self, incoming, reply, sent):
            self.deliveries.append((incoming.chat_id, incoming.user_id, sent))

    service = PreviewService()
    message = FailSecond("edita a.md")
    await OraculoMessageHandler(service)(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]
    assert len(message.replies) == 1
    assert service.deliveries == [(100, 1, False)]


# --- Contexto de respuesta (Req 2.1, 2.2, 2.3) ---


def test_reply_context_detecta_mensaje_del_bot():
    quoted = SimpleNamespace(from_user=SimpleNamespace(id=BOT_ID), text="respuesta previa")
    message = FakeReplyMessage("¿y un ejemplo?", reply_to_message=quoted)

    assert reply_context(message, BOT_ID) == (True, "respuesta previa")  # type: ignore[arg-type]


def test_reply_context_ignora_mensaje_de_tercero():
    quoted = SimpleNamespace(from_user=SimpleNamespace(id=999), text="mensaje ajeno")
    message = FakeReplyMessage("de acuerdo", reply_to_message=quoted)

    assert reply_context(message, BOT_ID) == (False, None)  # type: ignore[arg-type]


def test_reply_context_sin_cita():
    message = FakeReplyMessage("hola")

    assert reply_context(message, BOT_ID) == (False, None)  # type: ignore[arg-type]


async def test_handler_detecta_respuesta_al_bot():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    quoted = SimpleNamespace(from_user=SimpleNamespace(id=BOT_ID), text="respuesta previa")
    message = FakeReplyMessage("¿y un ejemplo?", reply_to_message=quoted)

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100),
        FakeContext(BOT_USERNAME),
    )

    incoming = service.received[0]
    assert incoming.replies_to_bot is True
    assert incoming.quoted_text == "respuesta previa"
    assert incoming.mentions_bot is False


async def test_handler_no_cita_mensajes_de_terceros():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    quoted = SimpleNamespace(from_user=SimpleNamespace(id=999), text="mensaje ajeno")
    message = make_message("@oraculo_bot dime algo", BOT_USERNAME, reply_to_message=quoted)

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100),
        FakeContext(BOT_USERNAME),
    )

    incoming = service.received[0]
    assert incoming.mentions_bot is True
    assert incoming.replies_to_bot is False
    assert incoming.quoted_text is None


async def test_handler_ignora_mensajes_de_bots():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("soy otro bot")

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100, user_id=777, is_bot=True),
        FakeContext(BOT_USERNAME),
    )

    assert service.received == []
    assert message.replies == []


# --- Nombre visible y comando de reinicio (Req 2.2, 4.1, 4.4) ---


async def test_handler_rellena_el_nombre_del_usuario():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("hola")

    await handler(  # type: ignore[arg-type]
        make_update(message, first_name="Ana"), FakeContext(BOT_USERNAME)
    )

    assert service.received[0].user_display_name == "Ana"


async def test_handler_usa_el_username_si_no_hay_nombre():
    service = RecordingService()
    handler = OraculoMessageHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("hola")

    await handler(  # type: ignore[arg-type]
        make_update(message, first_name="", username="ana_dev"),
        FakeContext(BOT_USERNAME),
    )

    assert service.received[0].user_display_name == "ana_dev"


async def test_help_handler_responde_con_la_lista_de_comandos():
    handler = OraculoCommandHandler(RecordingService())  # type: ignore[arg-type]
    message = FakeReplyMessage("/help")

    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert len(message.replies) == 1
    assert all(
        command in message.replies[0]
        for command in ("/new", "/agent", "/exitagent", "/help")
    )
    assert (
        "/agent — Activa el modo agente y permite elegir un proyecto "
        "(solo en privado)."
    ) in message.replies[0]
    assert "/exitagent — Sale del modo agente." in message.replies[0]


async def test_reset_handler_normaliza_y_confirma():
    service = RecordingService(reset_reply="sesión nueva")
    handler = OraculoResetHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("/new")

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-100),
        FakeContext(BOT_USERNAME),
    )

    assert len(service.reset_received) == 1
    incoming = service.reset_received[0]
    assert incoming.chat_type is ChatKind.GROUP
    assert incoming.chat_id == -100
    assert incoming.user_id == 1
    assert service.received == []  # no pasa por el flujo de mensajes
    assert message.replies == ["sesión nueva"]


async def test_reset_handler_ignora_bots():
    service = RecordingService()
    handler = OraculoResetHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("/new")

    await handler(  # type: ignore[arg-type]
        make_update(message, is_bot=True), FakeContext(BOT_USERNAME)
    )

    assert service.reset_received == []
    assert message.replies == []


async def test_reset_handler_tolera_fallo_de_envio():
    service = RecordingService()
    handler = OraculoResetHandler(service)  # type: ignore[arg-type]
    message = FailingReplyMessage("/new")

    await handler(make_update(message), FakeContext(BOT_USERNAME))  # type: ignore[arg-type]

    assert message.replies == []


async def test_reset_handler_no_responde_si_el_servicio_lo_ignora():
    service = RecordingService(reset_reply=None)
    handler = OraculoResetHandler(service)  # type: ignore[arg-type]
    message = FakeReplyMessage("/new")

    await handler(  # type: ignore[arg-type]
        make_update(message, chat_type="group", chat_id=-999),
        FakeContext(BOT_USERNAME),
    )

    assert message.replies == []


# --- Indicador de escritura (spec: indicador-de-escritura) ---


class SlowService(RecordingService):
    """Servicio cuyo manejo tarda lo indicado, simulando una consulta a la IA."""

    def __init__(self, reply: Optional[str] = "ok", delay_s: float = 0.05) -> None:
        super().__init__(reply=reply)
        self._delay_s = delay_s

    async def handle(self, message: IncomingMessage) -> Optional[str]:
        await asyncio.sleep(self._delay_s)
        return await super().handle(message)


async def test_handler_muestra_typing_si_el_servicio_tarda():
    context = FakeContext(BOT_USERNAME)
    handler = OraculoMessageHandler(  # type: ignore[arg-type]
        SlowService(reply="respuesta", delay_s=0.05),
        typing_start_delay_s=0.0,
        typing_refresh_s=10.0,
    )
    message = FakeReplyMessage("hola")

    await handler(make_update(message), context)  # type: ignore[arg-type]

    assert context.bot.actions == [(100, "typing")]
    assert message.replies == ["respuesta"]


async def test_handler_renueva_typing_si_tarda_mas_que_el_refresco():
    context = FakeContext(BOT_USERNAME)
    handler = OraculoMessageHandler(  # type: ignore[arg-type]
        SlowService(reply="ok", delay_s=0.25),
        typing_start_delay_s=0.0,
        typing_refresh_s=0.05,
    )

    await handler(make_update(FakeReplyMessage("hola")), context)  # type: ignore[arg-type]

    assert len(context.bot.actions) >= 2
    assert all(action == "typing" for _, action in context.bot.actions)


async def test_handler_no_muestra_typing_si_el_servicio_responde_rapido():
    context = FakeContext(BOT_USERNAME)
    handler = OraculoMessageHandler(  # type: ignore[arg-type]
        RecordingService(reply="ok"),
        typing_start_delay_s=5.0,
        typing_refresh_s=5.0,
    )
    message = FakeReplyMessage("hola")

    await handler(make_update(message), context)  # type: ignore[arg-type]

    assert context.bot.actions == []
    assert message.replies == ["ok"]


async def test_handler_deja_de_enviar_typing_al_terminar():
    context = FakeContext(BOT_USERNAME)
    handler = OraculoMessageHandler(  # type: ignore[arg-type]
        SlowService(reply="ok", delay_s=0.12),
        typing_start_delay_s=0.0,
        typing_refresh_s=0.05,
    )

    await handler(make_update(FakeReplyMessage("hola")), context)  # type: ignore[arg-type]
    count = len(context.bot.actions)
    assert count >= 1

    await asyncio.sleep(0.15)
    assert len(context.bot.actions) == count


async def test_handler_fallo_de_typing_no_interrumpe_la_respuesta():
    context = FakeContext(BOT_USERNAME, fail_send_action=True)
    handler = OraculoMessageHandler(  # type: ignore[arg-type]
        SlowService(reply="respuesta", delay_s=0.05),
        typing_start_delay_s=0.0,
        typing_refresh_s=0.05,
    )
    message = FakeReplyMessage("hola")

    await handler(make_update(message), context)  # type: ignore[arg-type]

    assert message.replies == ["respuesta"]
    assert context.bot.actions == [(100, "typing")]
