"""Tests del wiring de adjuntos en los handlers de Telegram."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from telegram import Chat, Message, MessageEntity
from telegram.error import TelegramError

from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.models import IncomingMessage
from oraculo.telegram.handlers import OraculoMessageHandler
from oraculo.telegram.session_locks import SessionUpdateLockPool


class FakeFile:
    def __init__(self, data: bytes | None, error: Exception | None = None) -> None:
        self._data = data
        self._error = error

    async def download_as_bytearray(self) -> bytearray:
        if self._error is not None:
            raise self._error
        return bytearray(self._data or b"")


class FakeBot:
    def __init__(self, files: dict[str, FakeFile]) -> None:
        self.username = "oraculo_bot"
        self.id = 42
        self._files = files
        self.actions: list[tuple[int, str]] = []

    async def get_file(self, file_id: str) -> FakeFile:
        return self._files[file_id]

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        self.actions.append((chat_id, action))


class FakeContext:
    def __init__(self, files: dict[str, FakeFile]) -> None:
        self.bot = FakeBot(files)


class RecordingService:
    def __init__(self, reply: str | None = "procesado") -> None:
        self.reply = reply
        self.attachments_calls: list[
            tuple[IncomingMessage, tuple[tuple[IncomingAttachment, bytes | None], ...]]
        ] = []
        self.plain: list[IncomingMessage] = []
        self.previews: list[tuple[str, bool]] = []

    async def handle_attachments(
        self,
        message: IncomingMessage,
        items: tuple[tuple[IncomingAttachment, bytes | None], ...],
    ) -> str | None:
        self.attachments_calls.append((message, items))
        return self.reply

    async def handle(self, message: IncomingMessage) -> str | None:
        self.plain.append(message)
        return self.reply

    async def handle_attachment(self, message: IncomingMessage) -> None:
        return None

    async def handle_reset(self, message: IncomingMessage) -> str | None:
        return None

    def preview_delivery(self, message: IncomingMessage, reply: str, sent: bool) -> None:
        self.previews.append((reply, sent))


class BlockingAttachmentService(RecordingService):
    def __init__(self) -> None:
        super().__init__(reply=None)
        self.started: asyncio.Queue[IncomingMessage] = asyncio.Queue()
        self.release = asyncio.Event()

    async def handle_attachments(
        self,
        message: IncomingMessage,
        items: tuple[tuple[IncomingAttachment, bytes | None], ...],
    ) -> str | None:
        self.attachments_calls.append((message, items))
        self.started.put_nowait(message)
        await self.release.wait()
        return None


class FakeMessage:
    def __init__(
        self,
        *,
        text: str | None = None,
        caption: str | None = None,
        media_group_id: str | None = None,
        file_id: str = "doc-1",
    ) -> None:
        self.text = text
        self.caption = caption
        self.entities = ()
        self.caption_entities = ()
        self.reply_to_message = None
        self.message_thread_id: int | None = None
        self.media_group_id = media_group_id
        self.document = SimpleNamespace(
            file_id=file_id, file_name="nota.txt",
            mime_type="text/plain", file_size=4,
        )
        self.photo = None
        self.audio = None
        self.video = None
        self.voice = None
        self.animation = None
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)

    def parse_caption_entity(self, entity: SimpleNamespace) -> str:
        start = entity.offset
        return (self.caption or "")[start:start + entity.length]


def document_message(
    *,
    text: str | None = None,
    caption: str | None = None,
    media_group_id: str | None = None,
    file_id: str = "doc-1",
) -> FakeMessage:
    return FakeMessage(
        text=text, caption=caption,
        media_group_id=media_group_id, file_id=file_id,
    )


def update_for(
    message: FakeMessage, chat_id: int = 100, chat_type: str = "private",
) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_user=SimpleNamespace(
            id=1, is_bot=False, first_name="Ana", username=None
        ),
    )


async def test_mensaje_con_adjunto_se_descarga_y_procesa():
    service = RecordingService()
    handler = OraculoMessageHandler(service)
    bot_files = {"doc-1": FakeFile(b"hola")}
    message = document_message(caption="resume")

    await handler(update_for(message), FakeContext(bot_files))

    assert len(service.attachments_calls) == 1
    incoming, items = service.attachments_calls[0]
    assert incoming.text == "resume"
    assert items == (
        (
            IncomingAttachment(
                file_id="doc-1", file_name="nota.txt",
                mime_type="text/plain", file_size=4, kind="document",
            ),
            b"hola",
        ),
    )
    assert message.replies == ["procesado"]


async def test_intake_de_adjuntos_del_mismo_privado_se_serializa() -> None:
    service = BlockingAttachmentService()
    handler = OraculoMessageHandler(
        service, session_update_locks=SessionUpdateLockPool(), typing_start_delay_s=30
    )
    context = FakeContext({
        "doc-1": FakeFile(b"uno"),
        "doc-2": FakeFile(b"dos"),
    })
    first = asyncio.create_task(
        handler(
            update_for(document_message(caption="primero", file_id="doc-1")),
            context,  # type: ignore[arg-type]
        )
    )
    first_incoming = await asyncio.wait_for(service.started.get(), timeout=1)
    second = asyncio.create_task(
        handler(
            update_for(document_message(caption="segundo", file_id="doc-2")),
            context,  # type: ignore[arg-type]
        )
    )
    await asyncio.sleep(0)

    assert first_incoming.text == "primero"
    assert service.started.empty()

    service.release.set()
    second_incoming = await asyncio.wait_for(service.started.get(), timeout=1)
    await asyncio.gather(first, second)

    assert second_incoming.text == "segundo"
    assert [incoming.text for incoming, _ in service.attachments_calls] == [
        "primero", "segundo"
    ]


async def test_descarga_fallida_se_omite_sin_abortar():
    service = RecordingService()
    handler = OraculoMessageHandler(service)
    bot_files = {"doc-1": FakeFile(None, error=TelegramError("red"))}
    message = document_message(caption="procesa")

    await handler(update_for(message), FakeContext(bot_files))

    _, items = service.attachments_calls[0]
    assert items == (
        (
            IncomingAttachment(
                file_id="doc-1", file_name="nota.txt",
                mime_type="text/plain", file_size=4, kind="document",
            ),
            None,
        ),
    )


async def test_caption_viaja_como_texto_del_mensaje():
    service = RecordingService()
    handler = OraculoMessageHandler(service)
    message = document_message(caption="resume el archivo")

    await handler(update_for(message), FakeContext({"doc-1": FakeFile(b"x")}))

    incoming, _ = service.attachments_calls[0]
    assert incoming.text == "resume el archivo"
    assert not service.plain


async def test_nota_de_voz_se_despacha_con_categoria_y_bytes_reales():
    service = RecordingService()
    handler = OraculoMessageHandler(service)
    message = document_message()
    message.document = None
    message.voice = SimpleNamespace(
        file_id="voice-1", file_name=None, mime_type=None, file_size=8,
    )

    await handler(update_for(message), FakeContext({"voice-1": FakeFile(b"OggSdata")}))

    incoming, items = service.attachments_calls[0]
    assert incoming.text is None
    assert len(items) == 1
    attachment, data = items[0]
    assert attachment.kind == "voice"
    assert attachment.file_name is None
    assert attachment.file_id == "voice-1"
    assert data == b"OggSdata"
    assert not service.plain


async def test_caption_y_mencion_de_voice_se_conservan_tras_normalizar():
    service = RecordingService()
    handler = OraculoMessageHandler(service)
    token = "@oraculo_bot"
    message = document_message(caption=f"{token} resume lo dicho")
    message.document = None
    message.voice = SimpleNamespace(
        file_id="voice-1", file_name=None, mime_type="audio/ogg", file_size=8,
    )
    message.caption_entities = (
        SimpleNamespace(
            type=MessageEntity.MENTION, offset=0, length=len(token),
        ),
    )

    await handler(update_for(message, chat_id=-100, chat_type="supergroup"),
                   FakeContext({"voice-1": FakeFile(b"OggSdata")}))

    incoming, items = service.attachments_calls[0]
    assert incoming.text == "resume lo dicho"
    assert incoming.mentions_bot is True
    assert incoming.chat_type.value == "group"
    assert items[0][0].kind == "voice"


async def test_album_se_agrupa_en_una_sola_interaccion():
    service = RecordingService()
    handler = OraculoMessageHandler(service, media_window_s=0.05)
    first = document_message(caption="analiza", media_group_id="album-1", file_id="doc-1")
    second = document_message(media_group_id="album-1", file_id="doc-2")
    bot_files = {"doc-1": FakeFile(b"uno"), "doc-2": FakeFile(b"dos")}
    context = FakeContext(bot_files)

    await handler(update_for(first), context)
    await handler(update_for(second), context)
    assert not service.attachments_calls

    await asyncio.sleep(0.12)

    assert len(service.attachments_calls) == 1
    incoming, items = service.attachments_calls[0]
    assert incoming.text == "analiza"
    assert [data for _, data in items] == [b"uno", b"dos"]


async def test_temporizador_de_un_album_no_vacia_otro_album_vencido(
    monkeypatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        "oraculo.telegram.media_groups.time.monotonic", lambda: clock[0]
    )
    real_sleep = asyncio.sleep
    timer_gates: asyncio.Queue[asyncio.Event] = asyncio.Queue()

    async def controlled_sleep(delay: float) -> None:
        if delay == 1.0:
            gate = asyncio.Event()
            timer_gates.put_nowait(gate)
            await gate.wait()
            return
        await real_sleep(delay)

    monkeypatch.setattr("oraculo.telegram.handlers.asyncio.sleep", controlled_sleep)
    service = RecordingService()
    handler = OraculoMessageHandler(
        service,
        media_window_s=1.0,
        session_update_locks=SessionUpdateLockPool(),
        typing_start_delay_s=30,
    )
    context = FakeContext({"doc-a": FakeFile(b"a"), "doc-b": FakeFile(b"b")})
    await handler(
        update_for(
            document_message(caption="a", media_group_id="album-a", file_id="doc-a"),
            chat_id=-100, chat_type="group",
        ),
        context,
    )
    await handler(
        update_for(
            document_message(caption="b", media_group_id="album-b", file_id="doc-b"),
            chat_id=-100, chat_type="group",
        ),
        context,
    )
    task_a = handler._album_tasks[(-100, "album-a")]
    task_b = handler._album_tasks[(-100, "album-b")]
    gate_a = await asyncio.wait_for(timer_gates.get(), timeout=1)
    gate_b = await asyncio.wait_for(timer_gates.get(), timeout=1)
    clock[0] = 2.0

    gate_a.set()
    await task_a

    assert len(service.attachments_calls) == 1
    assert service.attachments_calls[0][1][0][1] == b"a"

    gate_b.set()
    await task_b
    assert len(service.attachments_calls) == 2
    assert service.attachments_calls[1][1][0][1] == b"b"


async def test_tarea_diferida_de_album_toma_el_lock_de_su_conversacion() -> None:
    service = BlockingAttachmentService()
    handler = OraculoMessageHandler(
        service,
        media_window_s=0.02,
        session_update_locks=SessionUpdateLockPool(),
        typing_start_delay_s=30,
    )
    context = FakeContext({"doc-1": FakeFile(b"album")})
    album_message = document_message(
        caption="album", media_group_id="album-1", file_id="doc-1"
    )
    await handler(
        update_for(album_message, chat_id=-100, chat_type="group"), context
    )
    album_task = handler._album_tasks[(-100, "album-1")]
    await asyncio.wait_for(service.started.get(), timeout=1)

    next_message = document_message(caption="consulta")
    next_message.document = None
    next_update = asyncio.create_task(
        handler(update_for(next_message, chat_id=-100, chat_type="group"), context)
    )
    await asyncio.sleep(0)

    assert service.plain == []

    service.release.set()
    await asyncio.gather(album_task, next_update)
    assert len(service.plain) == 1


async def test_mencion_en_caption_se_detecta():
    from oraculo.telegram.handlers import mentions_bot

    token = "@oraculo_bot"
    message = Message(
        message_id=1,
        date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        chat=Chat(id=-100, type="supergroup"),
        caption=f"{token} resume esto",
        caption_entities=(MessageEntity(type=MessageEntity.MENTION, offset=0, length=len(token)),),
    )

    assert mentions_bot(message, "oraculo_bot") is True
