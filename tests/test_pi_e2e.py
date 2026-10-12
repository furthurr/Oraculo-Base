"""Tests de integración end-to-end del modo Pi.

Verifica el flujo completo: mensaje → PiModeService → eventos → respuesta.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace


from fakes import FakeProjectStore, enter_pi

from oraculo.application.audio import format_voice_prompt
from oraculo.application.pi_mode import PI_PREFIX, PiModeService
from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.models import IncomingMessage
from oraculo.domain.pi import (
    PiAccess,
    PiAgentPort,
    PiEvent,
    PiSessionState,
    PiToolEvent,
)
from oraculo.telegram.handlers import OraculoMessageHandler


class E2EPiAgent:
    """Agente Pi simulado que reproduce un flujo realista."""

    def __init__(self) -> None:
        self._event_callbacks: list = []
        self._tool_callbacks: list = []
        self.prompts: list[str] = []
        self.closed = False

    async def send_prompt(self, text, images=None):
        self.prompts.append(text)
        asyncio.ensure_future(self._simulate_response())

    async def _simulate_response(self):
        await asyncio.sleep(0.01)
        self._emit_tool("read")
        await asyncio.sleep(0.01)
        self._emit_text("Analizando el proyecto...")
        await asyncio.sleep(0.01)
        self._emit_text(" Encontré 3 archivos.")
        await asyncio.sleep(0.01)
        self._emit_end("Analizando el proyecto... Encontré 3 archivos.")

    def _emit_text(self, text):
        for cb in self._event_callbacks:
            cb(PiEvent(type="text_delta", content=text))

    def _emit_end(self, text):
        for cb in self._event_callbacks:
            cb(PiEvent(type="message_end", content=text))
        for cb in self._event_callbacks:
            cb(PiEvent(type="settled", content=""))

    def _emit_tool(self, name):
        for cb in self._tool_callbacks:
            cb(PiToolEvent(type="tool_start", tool_name=name))

    async def send_steer(self, text, images=None):
        pass

    async def abort(self):
        pass

    async def new_session(self):
        pass

    async def get_state(self):
        return {"isStreaming": False}

    def on_event(self, cb):
        self._event_callbacks.append(cb)

    def on_tool_event(self, cb):
        self._tool_callbacks.append(cb)

    async def close(self):
        self.closed = True


async def collect(service, user_id, text):
    return [e async for e in service.handle(user_id, text)]


class TestEndToEnd:
    async def test_full_conversation_flow(self) -> None:
        agent = E2EPiAgent()

        async def factory(user_id, cwd):
            return agent

        service = PiModeService(agent_factory=factory, project_store=FakeProjectStore())

        events = await enter_pi(service, 1)
        assert any(e.type == "response" for e in events)
        assert service.get_state(1) is PiSessionState.PI_ACTIVO

        events = await collect(service, 1, "analiza el proyecto")

        tool_notices = [e for e in events if e.type == "tool_notice"]
        text_deltas = [e for e in events if e.type == "text_delta"]
        text_finals = [e for e in events if e.type == "text_final"]

        assert len(tool_notices) == 1
        assert "🤖💻" in tool_notices[0].content
        assert len(text_deltas) == 2
        assert len(text_finals) == 1
        assert "Analizando" in text_finals[0].content

    async def test_exit_after_conversation(self) -> None:
        agent = E2EPiAgent()

        async def factory(user_id, cwd):
            return agent

        service = PiModeService(agent_factory=factory, project_store=FakeProjectStore())

        await enter_pi(service, 1)
        await collect(service, 1, "hola")
        await collect(service, 1, "/exitagent")

        assert service.get_state(1) is PiSessionState.NORMAL
        assert agent.closed

    async def test_multiple_users_isolated(self) -> None:
        agents = {}

        async def factory(user_id, cwd):
            agent = E2EPiAgent()
            agents[user_id] = agent
            return agent

        service = PiModeService(agent_factory=factory, project_store=FakeProjectStore())

        await enter_pi(service, 1)
        await enter_pi(service, 2)

        await collect(service, 1, "hola")
        await collect(service, 2, "adiós")

        assert agents[1].prompts == ["hola"]
        assert agents[2].prompts == ["adiós"]
        assert agents[1] is not agents[2]

    async def test_new_session_resets(self) -> None:
        agent = E2EPiAgent()

        async def factory(user_id, cwd):
            return agent

        service = PiModeService(agent_factory=factory, project_store=FakeProjectStore())

        await enter_pi(service, 1)
        await collect(service, 1, "primera")
        events = await collect(service, 1, "/new")

        assert any("reiniciada" in e.content for e in events)
        assert service.get_state(1) is PiSessionState.PI_ACTIVO


class VoiceE2EPiAgent(E2EPiAgent):
    async def _simulate_response(self) -> None:
        await asyncio.sleep(0.01)
        self._emit_end("He recibido la transcripción.")


class VoiceE2ETranscriber:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, data: bytes, *, mime_type: str) -> str:
        self.calls.append((data, mime_type))
        return self.transcript


class VoiceAccessService:
    def __init__(self) -> None:
        self.normal_attachment_calls: list[
            tuple[IncomingMessage, tuple[tuple[IncomingAttachment, bytes | None], ...]]
        ] = []

    def pi_access(self, message: IncomingMessage) -> PiAccess:
        del message
        return PiAccess(allowed=True, reply=None)

    async def handle_attachments(
        self,
        message: IncomingMessage,
        items: Sequence[tuple[IncomingAttachment, bytes | None]],
    ) -> str | None:
        self.normal_attachment_calls.append((message, tuple(items)))
        return None


class VoiceTelegramMessage:
    def __init__(self, *, file_id: str, size: int, caption: str) -> None:
        self.text: None = None
        self.caption = caption
        self.voice = SimpleNamespace(
            file_id=file_id,
            file_name=None,
            mime_type="audio/ogg",
            file_size=size,
        )
        self.entities: tuple[object, ...] = ()
        self.caption_entities: tuple[object, ...] = ()
        self.reply_to_message: None = None
        self.message_thread_id: None = None
        self.replies: list[str] = []

    def parse_entity(self, entity: object) -> str:
        del entity
        return ""

    def parse_caption_entity(self, entity: object) -> str:
        del entity
        return ""

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class VoiceTelegramBot:
    def __init__(self, file_id: str, audio: bytes) -> None:
        self.username = "oraculo_bot"
        self.id = 424242
        self.file_id = file_id
        self.audio = audio
        self.requested_file_ids: list[str] = []

    async def get_file(self, file_id: str) -> object:
        self.requested_file_ids.append(file_id)
        assert file_id == self.file_id
        audio = self.audio

        class Download:
            async def download_as_bytearray(self) -> bytearray:
                return bytearray(audio)

        return Download()


class TestVoiceToAgentEndToEnd:
    async def test_telegram_voice_is_transcribed_and_only_text_reaches_pi(self) -> None:
        audio = b"OggSnota de voz en espanol"
        caption = "Resume mi petición"
        transcript = "Revisa el README"
        agent = VoiceE2EPiAgent()
        transcriber = VoiceE2ETranscriber(transcript)

        async def factory(user_id: int, cwd: str) -> PiAgentPort:
            del user_id, cwd
            return agent  # type: ignore[return-value]

        pi_service = PiModeService(
            agent_factory=factory,
            project_store=FakeProjectStore(),
            audio_transcriber=transcriber,
        )
        await enter_pi(pi_service, 1)
        access_service = VoiceAccessService()
        handler = OraculoMessageHandler(
            service=access_service,  # type: ignore[arg-type]
            pi_mode_service=pi_service,
        )
        message = VoiceTelegramMessage(
            file_id="voice-e2e", size=len(audio), caption=caption,
        )
        bot = VoiceTelegramBot("voice-e2e", audio)
        update = SimpleNamespace(
            effective_message=message,
            effective_chat=SimpleNamespace(id=100, type="private"),
            effective_user=SimpleNamespace(
                id=1, is_bot=False, first_name="Ana", username=None,
            ),
        )

        await handler(update, SimpleNamespace(bot=bot))  # type: ignore[arg-type]

        assert bot.requested_file_ids == ["voice-e2e"]
        assert transcriber.calls == [(audio, "audio/ogg")]
        assert agent.prompts == [format_voice_prompt(caption, transcript)]
        assert audio.decode() not in agent.prompts[0]
        assert access_service.normal_attachment_calls == []
        assert message.replies == [f"{PI_PREFIX} He recibido la transcripción."]