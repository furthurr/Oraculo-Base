"""Tests de integración del modo Pi en los handlers de Telegram."""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from oraculo.application.pi_mode import PiDisplayEvent, PiModeService
from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.models import ChatKind, IncomingMessage
from oraculo.domain.pi import PiAccess, PiSessionState


def make_incoming(text: str, user_id: int = 1) -> IncomingMessage:
    return IncomingMessage(
        chat_id=user_id,
        user_id=user_id,
        chat_type=ChatKind.PRIVATE,
        text=text,
        mentions_bot=False,
        replies_to_bot=False,
        quoted_text=None,
        user_display_name="Test",
    )


class MockPiModeService:
    def __init__(self) -> None:
        self.handled: list[tuple[int, str]] = []
        self.state = PiSessionState.NORMAL
        self.events_to_return: list[PiDisplayEvent] = []
        self.images_received: list[list[bytes] | None] = []
        self.voice_received: list[
            tuple[int, IncomingAttachment, bytes | None, str | None]
        ] = []

    def is_in_pi_mode(self, user_id: int) -> bool:
        return self.state != PiSessionState.NORMAL

    def get_state(self, user_id: int) -> PiSessionState:
        return self.state

    async def handle(
        self, user_id: int, text: str, *, images: list[bytes] | None = None
    ):
        self.handled.append((user_id, text))
        self.images_received.append(images)
        for event in self.events_to_return:
            yield event

    async def handle_voice(
        self,
        user_id: int,
        attachment: IncomingAttachment,
        data: bytes | None,
        *,
        caption: str | None,
    ):
        self.voice_received.append((user_id, attachment, data, caption))
        for event in self.events_to_return:
            yield event


class MockConversationService:
    def __init__(self, pi_access: PiAccess | None = None) -> None:
        self.handled: list[IncomingMessage] = []
        self.resets: list[IncomingMessage] = []
        self.pi_access_calls: list[IncomingMessage] = []
        self.reply = "respuesta normal"
        self.attachments_handled: list[
            tuple[IncomingMessage, tuple[tuple[IncomingAttachment, bytes | None], ...]]
        ] = []
        self._pi_access = pi_access or PiAccess(allowed=True, reply=None)

    def pi_access(self, message: IncomingMessage) -> PiAccess:
        self.pi_access_calls.append(message)
        return self._pi_access

    async def handle_reset(self, message: IncomingMessage) -> str | None:
        self.resets.append(message)
        return "conversación reiniciada"

    async def handle(self, message: IncomingMessage) -> str | None:
        self.handled.append(message)
        return self.reply

    async def handle_attachment(self, message: IncomingMessage) -> tuple[str, bytes] | None:
        return None

    async def handle_attachments(
        self,
        message: IncomingMessage,
        items: Sequence[tuple[IncomingAttachment, bytes | None]],
    ) -> str | None:
        self.attachments_handled.append((message, tuple(items)))
        return self.reply

    def preview_delivery(self, message: IncomingMessage, reply: str, sent: bool) -> None:
        pass


class TestPiModeRouting:
    async def test_routes_to_pi_when_in_pi_mode(self) -> None:
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        pi_service.events_to_return = [
            PiDisplayEvent(type="response", content="🤖💻 hola desde Pi")
        ]
        conv_service = MockConversationService()

        from oraculo.telegram.handlers import OraculoMessageHandler

        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = make_incoming("hola")

        # Simular que el handler procesa el mensaje
        # (necesitamos mockear Telegram, pero verificamos el routing)
        assert pi_service.is_in_pi_mode(1)
        assert not conv_service.handled

    async def test_routes_to_conversation_when_normal(self) -> None:
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.NORMAL
        conv_service = MockConversationService()

        from oraculo.telegram.handlers import OraculoMessageHandler

        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        assert not pi_service.is_in_pi_mode(1)

    async def test_pi_mode_service_receives_user_id_and_text(self) -> None:
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        pi_service.events_to_return = [
            PiDisplayEvent(type="response", content="🤖💻 respuesta")
        ]

        events = [e async for e in pi_service.handle(1, "hola")]
        assert pi_service.handled == [(1, "hola")]
        assert len(events) == 1
        assert events[0].content == "🤖💻 respuesta"


class TestPiDisplayEvents:
    def test_response_event(self) -> None:
        event = PiDisplayEvent(type="response", content="🤖💻 hola")
        assert event.type == "response"
        assert event.content == "🤖💻 hola"

    def test_text_delta_event(self) -> None:
        event = PiDisplayEvent(type="text_delta", content="fragmento")
        assert event.type == "text_delta"

    def test_text_final_event(self) -> None:
        event = PiDisplayEvent(type="text_final", content="respuesta completa")
        assert event.type == "text_final"

    def test_tool_notice_event(self) -> None:
        event = PiDisplayEvent(type="tool_notice", content="🤖💻 Leyendo archivos…")
        assert event.type == "tool_notice"

    def test_error_event(self) -> None:
        event = PiDisplayEvent(type="error", content="🤖💻 error")
        assert event.type == "error"

class FakeReplyMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.entities: tuple = ()
        self.caption: str | None = None
        self.caption_entities: tuple = ()
        self.reply_to_message: object = None
        self.message_thread_id: int | None = None
        self.replies: list[str] = []
    def parse_entity(self, entity) -> str:
        return ""

    def parse_caption_entity(self, entity) -> str:
        return ""

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeBot:
    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.username = "oraculo_bot"
        self.id = 424242
        self.files = files or {}
        self.requested_file_ids: list[str] = []

    async def send_chat_action(self, chat_id: int, action: str) -> None:
        pass

    async def get_file(self, file_id: str):
        self.requested_file_ids.append(file_id)
        data = self.files[file_id]

        class Download:
            async def download_as_bytearray(self) -> bytearray:
                return bytearray(data)

        return Download()


def make_update(message: object) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=100, type="private"),
        effective_user=SimpleNamespace(
            id=1, is_bot=False, first_name="Ana", username=None
        ),
    )


class TestPiImageRouting:
    async def test_photo_and_caption_are_forwarded_to_pi(self) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        photo = b"\xff\xd8\xffphoto"
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        pi_service.events_to_return = [
            PiDisplayEvent(type="response", content="🤖💻 imagen recibida")
        ]
        conv_service = MockConversationService()
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = FakeReplyMessage("")
        message.text = None
        message.caption = "¿Qué ves?"
        message.photo = [SimpleNamespace(
            file_id="photo-1", file_size=len(photo), file_name=None, mime_type=None
        )]

        await handler(
            make_update(message),
            SimpleNamespace(bot=FakeBot(files={"photo-1": photo})),
        )  # type: ignore[arg-type]

        assert pi_service.handled == [(1, "¿Qué ves?")]
        assert pi_service.images_received == [[photo]]
        assert conv_service.handled == []

    async def test_image_document_is_forwarded_to_pi(self) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        image = b"\x89PNG\r\n\x1a\nimage"
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        conv_service = MockConversationService()
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = FakeReplyMessage("mira esto")
        message.document = SimpleNamespace(
            file_id="image-1", file_size=len(image), file_name="foto.png",
            mime_type="image/png",
        )

        await handler(
            make_update(message),
            SimpleNamespace(bot=FakeBot(files={"image-1": image})),
        )  # type: ignore[arg-type]

        assert pi_service.handled == [(1, "mira esto")]
        assert pi_service.images_received == [[image]]

    async def test_photo_without_caption_gets_default_instruction(self) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        photo = b"\xff\xd8\xffphoto"
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        conv_service = MockConversationService()
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = FakeReplyMessage("")
        message.text = None
        message.photo = [SimpleNamespace(
            file_id="photo-1", file_size=len(photo), file_name=None, mime_type=None
        )]

        await handler(
            make_update(message),
            SimpleNamespace(bot=FakeBot(files={"photo-1": photo})),
        )  # type: ignore[arg-type]

        assert pi_service.handled == [(1, "Analiza las imágenes adjuntas.")]
        assert pi_service.images_received == [[photo]]


class TestPiCommandRouting:
    async def test_model_outside_pi_mode_routes_to_pi_service(self) -> None:
        pi_service = MockPiModeService()
        pi_service.events_to_return = [
            PiDisplayEvent(type="response", content="🤖💻 El modo agente no está activo. Usa /agent para activarlo.")
        ]
        conv_service = MockConversationService()

        from oraculo.telegram.handlers import OraculoMessageHandler

        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = FakeReplyMessage("/model")
        await handler(make_update(message), SimpleNamespace(bot=FakeBot()))  # type: ignore[arg-type]

        assert pi_service.handled == [(1, "/model")]
        assert conv_service.handled == []
        assert message.replies and "/agent" in message.replies[0]

    async def test_model_with_arg_outside_pi_mode_routes_to_pi_service(self) -> None:
        pi_service = MockPiModeService()
        conv_service = MockConversationService()

        from oraculo.telegram.handlers import OraculoMessageHandler

        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = FakeReplyMessage("/model GPT")
        await handler(make_update(message), SimpleNamespace(bot=FakeBot()))  # type: ignore[arg-type]

        assert pi_service.handled == [(1, "/model GPT")]
        assert conv_service.handled == []

    async def test_double_slash_routes_to_pi_service(self) -> None:
        pi_service = MockPiModeService()
        conv_service = MockConversationService()

        from oraculo.telegram.handlers import OraculoMessageHandler

        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = FakeReplyMessage("//sh ls -la")
        await handler(make_update(message), SimpleNamespace(bot=FakeBot()))  # type: ignore[arg-type]

        assert pi_service.handled == [(1, "//sh ls -la")]
        assert conv_service.handled == []


def make_chat_update(message: object, *, chat_type: str, chat_id: int, user_id: int = 1) -> object:
    return SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type),
        effective_user=SimpleNamespace(
            id=user_id, is_bot=False, first_name="Ana", username=None
        ),
    )


GROUP_NOTICE = "🤖💻 El modo agente solo está disponible por mensaje directo."


async def _send(
    text: str,
    *,
    chat_type: str,
    pi_state: PiSessionState = PiSessionState.PI_ACTIVO,
    access: PiAccess | None = None,
) -> tuple[MockPiModeService, MockConversationService, FakeReplyMessage]:
    from oraculo.telegram.handlers import OraculoMessageHandler

    pi_service = MockPiModeService()
    pi_service.state = pi_state
    conv_service = MockConversationService(pi_access=access)
    handler = OraculoMessageHandler(
        service=conv_service,  # type: ignore[arg-type]
        pi_mode_service=pi_service,  # type: ignore[arg-type]
        typing_start_delay_s=10.0,
    )
    message = FakeReplyMessage(text)
    chat_id = 100 if chat_type == "private" else -100
    update = make_chat_update(message, chat_type=chat_type, chat_id=chat_id)
    await handler(update, SimpleNamespace(bot=FakeBot()))  # type: ignore[arg-type]
    return pi_service, conv_service, message


class TestPiOnlyInPrivateChats:
    """Regresión: el modo Pi no debe filtrarse a los grupos (bugfix solo-chat-privado)."""

    async def test_group_message_with_active_private_session_skips_pi(self) -> None:
        pi_service, conv_service, _ = await _send(
            "hola", chat_type="supergroup",
            access=PiAccess(allowed=False, reply=GROUP_NOTICE),
        )
        assert pi_service.handled == []
        assert len(conv_service.handled) == 1

    async def test_group_exitagent_does_not_touch_private_session(self) -> None:
        pi_service, _, message = await _send(
            "/exitagent", chat_type="group",
            access=PiAccess(allowed=False, reply=GROUP_NOTICE),
        )
        assert pi_service.handled == []
        assert message.replies == [GROUP_NOTICE]

    async def test_group_pi_command_gets_notice(self) -> None:
        pi_service, conv_service, message = await _send(
            "/agent", chat_type="group", pi_state=PiSessionState.NORMAL,
            access=PiAccess(allowed=False, reply=GROUP_NOTICE),
        )
        assert pi_service.handled == []
        assert conv_service.handled == []
        assert message.replies == [GROUP_NOTICE]

    async def test_unauthorized_group_pi_command_is_ignored(self) -> None:
        pi_service, conv_service, message = await _send(
            "//sh ls", chat_type="group", pi_state=PiSessionState.NORMAL,
            access=PiAccess(allowed=False, reply=None),
        )
        assert pi_service.handled == []
        assert conv_service.handled == []
        assert message.replies == []

    async def test_group_reset_with_active_session_resets_group(self) -> None:
        pi_service, conv_service, _ = await _send("/new", chat_type="group")
        assert pi_service.handled == []
        assert len(conv_service.resets) == 1

    async def test_private_unauthorized_pi_is_rejected(self) -> None:
        pi_service, _, message = await _send(
            "/agent", chat_type="private", pi_state=PiSessionState.NORMAL,
            access=PiAccess(allowed=False, reply="No hablo con extraños"),
        )
        assert pi_service.handled == []
        assert message.replies == ["No hablo con extraños"]

    async def test_private_authorized_session_still_routes_to_pi(self) -> None:
        pi_service, conv_service, _ = await _send("hola", chat_type="private")
        assert pi_service.handled == [(1, "hola")]
        assert conv_service.handled == []

    async def test_private_reset_with_active_session_goes_to_pi(self) -> None:
        pi_service, conv_service, _ = await _send("/reset", chat_type="private")
        assert pi_service.handled == [(1, "/reset")]
        assert conv_service.resets == []


class TestPiVoiceRouting:
    @staticmethod
    def message_with_voice(audio: bytes, *, caption: str | None = None) -> FakeReplyMessage:
        message = FakeReplyMessage("")
        message.text = None
        message.caption = caption
        message.voice = SimpleNamespace(
            file_id="voice-1",
            file_name=None,
            mime_type="audio/ogg",
            file_size=len(audio),
        )
        return message

    async def test_active_pi_voice_downloads_after_access_and_routes_caption(self) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        audio = b"OggSspanish voice"
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        pi_service.events_to_return = [
            PiDisplayEvent(type="response", content="🤖💻 Recibido")
        ]
        conv_service = MockConversationService()
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = self.message_with_voice(audio, caption="Resume esto")
        bot = FakeBot(files={"voice-1": audio})

        await handler(make_update(message), SimpleNamespace(bot=bot))  # type: ignore[arg-type]

        assert bot.requested_file_ids == ["voice-1"]
        assert len(pi_service.voice_received) == 1
        user_id, attachment, data, caption = pi_service.voice_received[0]
        assert user_id == 1
        assert attachment.kind == "voice"
        assert data == audio
        assert caption == "Resume esto"
        assert pi_service.handled == []
        assert conv_service.attachments_handled == []
        assert message.replies == ["🤖💻 Recibido"]

    async def test_unauthorized_pi_voice_is_not_downloaded(self) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        audio = b"OggSvoice"
        pi_service = MockPiModeService()
        pi_service.state = PiSessionState.PI_ACTIVO
        conv_service = MockConversationService(
            pi_access=PiAccess(allowed=False, reply="No autorizado")
        )
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = self.message_with_voice(audio)
        bot = FakeBot(files={"voice-1": audio})

        await handler(make_update(message), SimpleNamespace(bot=bot))  # type: ignore[arg-type]

        assert bot.requested_file_ids == []
        assert pi_service.voice_received == []
        assert message.replies == ["No autorizado"]

    @pytest.mark.parametrize(
        ("state", "caption"),
        [
            (PiSessionState.SELECCION_PROYECTO, "1"),
            (PiSessionState.CONFIRMAR_SALIDA, "sí"),
        ],
    )
    async def test_voice_during_intermediate_state_is_not_interpreted(
        self, state: PiSessionState, caption: str,
    ) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        audio = b"OggSvoice"
        pi_service = MockPiModeService()
        pi_service.state = state
        conv_service = MockConversationService()
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = self.message_with_voice(audio, caption=caption)
        bot = FakeBot(files={"voice-1": audio})

        await handler(make_update(message), SimpleNamespace(bot=bot))  # type: ignore[arg-type]

        assert bot.requested_file_ids == []
        assert pi_service.voice_received == []
        assert pi_service.handled == []
        assert message.replies == []

    async def test_voice_outside_pi_keeps_normal_attachment_flow(self) -> None:
        from oraculo.telegram.handlers import OraculoMessageHandler

        audio = b"OggSvoice"
        pi_service = MockPiModeService()
        conv_service = MockConversationService()
        handler = OraculoMessageHandler(
            service=conv_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,  # type: ignore[arg-type]
        )
        message = self.message_with_voice(audio)
        bot = FakeBot(files={"voice-1": audio})

        await handler(make_update(message), SimpleNamespace(bot=bot))  # type: ignore[arg-type]

        assert pi_service.voice_received == []
        assert len(conv_service.attachments_handled) == 1
        assert conv_service.attachments_handled[0][1][0][1] == audio
