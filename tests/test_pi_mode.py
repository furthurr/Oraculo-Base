"""Tests del servicio de modo Pi: máquina de estados, streaming y comandos."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from oraculo.domain.attachments import IncomingAttachment
from oraculo.domain.errors import AudioTranscriptionError
from oraculo.domain.pi import (
    PiAgentPort,
    PiEvent,
    PiModeError,
    PiSessionState,
    PiToolEvent,
)
from fakes import FakeProjectStore, enter_pi

from oraculo.application.pi_mode import PI_PREFIX, PiDisplayEvent, PiModeService


async def collect(service: PiModeService, user_id: int, text: str) -> list[PiDisplayEvent]:
    return [e async for e in service.handle(user_id, text)]


class MockPiAgent:
    """Agente Pi simulado para tests."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.prompt_images: list[list[bytes] | None] = []
        self.steers: list[str] = []
        self.steer_images: list[list[bytes] | None] = []
        self.aborted = False
        self.new_session_called = False
        self.closed = False
        self._event_callbacks: list = []
        self._tool_callbacks: list = []
        self._is_streaming = False
        self.auto_respond = True
        self.models: list[dict] = [
            {"id": "claude-x", "name": "Claude X", "provider": "anthropic"},
            {"id": "gpt-5", "name": "GPT-5", "provider": "openai"},
        ]
        self.commands: list[dict] = [
            {"name": "fix-tests", "description": "Fix", "source": "prompt"},
        ]
        self.current_model: dict | None = dict(self.models[0])
        self.set_model_calls: list[tuple[str, str]] = []
        self.bash_calls: list[str] = []
        self.bash_result: dict = {
            "output": "total 0",
            "exitCode": 0,
            "cancelled": False,
            "truncated": False,
        }

    async def send_prompt(self, text: str, images=None) -> None:
        self.prompts.append(text)
        self.prompt_images.append(images)
        self._is_streaming = True
        if self.auto_respond:
            asyncio.ensure_future(self._auto_end())

    async def send_steer(self, text: str, images=None) -> None:
        self.steers.append(text)
        self.steer_images.append(images)

    async def abort(self) -> None:
        self.aborted = True
        self._is_streaming = False

    async def new_session(self) -> None:
        self.new_session_called = True

    async def get_state(self) -> dict:
        state: dict = {"isStreaming": self._is_streaming}
        if self.current_model is not None:
            state["model"] = dict(self.current_model)
        return state

    async def set_model(self, provider: str, model_id: str) -> dict:
        self.set_model_calls.append((provider, model_id))
        for model in self.models:
            if model["id"] == model_id and model["provider"] == provider:
                self.current_model = dict(model)
                return dict(model)
        raise PiModeError("rpc_error", user_message="Modelo no encontrado")

    async def get_available_models(self) -> list[dict]:
        return [dict(m) for m in self.models]

    async def get_commands(self) -> list[dict]:
        return [dict(c) for c in self.commands]

    async def bash(self, command: str) -> dict:
        self.bash_calls.append(command)
        return dict(self.bash_result)

    def on_event(self, callback) -> None:
        self._event_callbacks.append(callback)

    def on_tool_event(self, callback) -> None:
        self._tool_callbacks.append(callback)

    async def close(self) -> None:
        self.closed = True

    def emit_text(self, text: str) -> None:
        for cb in self._event_callbacks:
            cb(PiEvent(type="text_delta", content=text))

    def emit_end(self, text: str, *, settle: bool = True) -> None:
        """`message_end` y, como Pi real, `settled` al terminar el turno (Req 3.1)."""
        self._is_streaming = not settle
        for cb in self._event_callbacks:
            cb(PiEvent(type="message_end", content=text))
        if settle:
            self.emit_settled()

    def emit_settled(self) -> None:
        self._is_streaming = False
        for cb in self._event_callbacks:
            cb(PiEvent(type="settled", content=""))

    def emit_tool(self, name: str) -> None:
        for cb in self._tool_callbacks:
            cb(PiToolEvent(type="tool_start", tool_name=name))

    async def _auto_end(self) -> None:
        await asyncio.sleep(0.01)
        self.emit_end("respuesta")


class FakeAudioTranscriber:
    def __init__(
        self,
        transcript: str = "Hola, ¿puedes ayudarme?",
        error: AudioTranscriptionError | None = None,
    ) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, data: bytes, *, mime_type: str) -> str:
        self.calls.append((data, mime_type))
        if self.error is not None:
            raise self.error
        return self.transcript


def make_service(
    audio_transcriber: FakeAudioTranscriber | None = None,
) -> tuple[PiModeService, dict[int, MockPiAgent]]:
    agents: dict[int, MockPiAgent] = {}

    async def factory(user_id: int, cwd: str) -> PiAgentPort:
        agent = MockPiAgent()
        agents[user_id] = agent
        return agent  # type: ignore[return-value]

    if audio_transcriber is None:
        service = PiModeService(agent_factory=factory, project_store=FakeProjectStore())
    else:
        service = PiModeService(
            agent_factory=factory,
            project_store=FakeProjectStore(),
            audio_transcriber=audio_transcriber,
        )
    return service, agents


class TestStateMachine:
    async def test_enters_pi_mode(self) -> None:
        service, agents = make_service()
        events = await enter_pi(service, 1)
        assert service.get_state(1) is PiSessionState.PI_ACTIVO
        assert any("🤖💻" in e.content for e in events)

    async def test_exits_pi_mode(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/exitagent")
        assert service.get_state(1) is PiSessionState.NORMAL

    async def test_exit_with_confirmation_when_streaming(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1]._is_streaming = True
        events = await collect(service, 1, "/exitagent")
        assert service.get_state(1) is PiSessionState.CONFIRMAR_SALIDA
        assert any("sí" in e.content.lower() and "no" in e.content.lower() for e in events)

    async def test_confirm_exit_with_yes(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1]._is_streaming = True
        await collect(service, 1, "/exitagent")
        events = await collect(service, 1, "sí")
        assert service.get_state(1) is PiSessionState.NORMAL

    async def test_confirm_exit_with_no(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1]._is_streaming = True
        await collect(service, 1, "/exitagent")
        events = await collect(service, 1, "no")
        assert service.get_state(1) is PiSessionState.PI_ACTIVO

    async def test_stoppi_aborts_work(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1]._is_streaming = True
        events = await collect(service, 1, "/stoppi")
        assert agents[1].aborted
        assert service.get_state(1) is PiSessionState.PI_ACTIVO

    async def test_new_resets_session(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/new")
        assert agents[1].new_session_called
        assert service.get_state(1) is PiSessionState.PI_ACTIVO

    async def test_reset_resets_session(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/reset")
        assert agents[1].new_session_called

    async def test_pi_when_already_active(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/agent")
        assert service.get_state(1) is PiSessionState.PI_ACTIVO
        assert any("Ya estás en modo agente" in e.content for e in events)

    async def test_normal_mode_ignores_messages(self) -> None:
        service, agents = make_service()
        events = await collect(service, 1, "hola")
        assert len(events) == 0


class TestMessageRouting:
    async def test_message_sent_to_pi(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "hola mundo")
        assert agents[1].prompts == ["hola mundo"]

    async def test_image_is_sent_with_prompt_to_pi(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        image = b"image-data"
        _ = [
            event async for event in service.handle(1, "describe", images=[image])
        ]
        assert agents[1].prompts == ["describe"]
        assert agents[1].prompt_images == [[image]]

    async def test_streaming_text_delta(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].auto_respond = False

        async def simulate():
            await asyncio.sleep(0.05)
            agents[1].emit_text("Hola ")
            await asyncio.sleep(0.05)
            agents[1].emit_text("mundo")
            await asyncio.sleep(0.05)
            agents[1].emit_end("Hola mundo")

        task = asyncio.create_task(simulate())
        events = await collect(service, 1, "di hola")
        await task

        deltas = [e for e in events if e.type == "text_delta"]
        finals = [e for e in events if e.type == "text_final"]
        assert len(deltas) == 2
        assert len(finals) == 1
        assert finals[0].content == "Hola mundo"

    async def test_tool_notice(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].auto_respond = False

        async def simulate():
            await asyncio.sleep(0.05)
            agents[1].emit_tool("read")
            await asyncio.sleep(0.05)
            agents[1].emit_end("listo")

        task = asyncio.create_task(simulate())
        events = await collect(service, 1, "lee el archivo")
        await task

        notices = [e for e in events if e.type == "tool_notice"]
        assert len(notices) == 1
        assert notices[0].content == "🤖💻 Leyendo archivos…"

    @pytest.mark.parametrize(
        ("tool_name", "expected"),
        [
            ("bash", "🤖💻 Ejecutando comandos…"),
            ("edit", "🤖💻 El agente está editando archivos…"),
        ],
    )
    async def test_tool_notice_copy(self, tool_name: str, expected: str) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].auto_respond = False

        async def simulate() -> None:
            await asyncio.sleep(0.01)
            agents[1].emit_tool(tool_name)
            await asyncio.sleep(0.01)
            agents[1].emit_end("listo")

        task = asyncio.create_task(simulate())
        events = await collect(service, 1, "continúa")
        await task
        notices = [event for event in events if event.type == "tool_notice"]
        assert [event.content for event in notices] == [expected]

    async def test_steer_when_streaming(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1]._is_streaming = True
        events = await collect(service, 1, "cambia el rumbo")
        assert agents[1].steers == ["cambia el rumbo"]

    async def test_image_is_included_in_steer_while_streaming(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1]._is_streaming = True
        image = b"image-data"
        _ = [
            event async for event in service.handle(1, "mira", images=[image])
        ]
        assert agents[1].steers == ["mira"]
        assert agents[1].steer_images == [[image]]


class TestErrorHandling:
    async def test_error_emits_error_event(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].auto_respond = False

        async def simulate():
            await asyncio.sleep(0.05)
            for cb in agents[1]._event_callbacks:
                cb(PiEvent(type="error", content="model_error: fallo"))

        task = asyncio.create_task(simulate())
        events = await collect(service, 1, "hola")
        await task

        errors = [e for e in events if e.type == "error"]
        assert len(errors) == 1
        assert "🤖💻" in errors[0].content

    async def test_stays_in_pi_mode_after_error(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].auto_respond = False

        async def simulate():
            await asyncio.sleep(0.05)
            for cb in agents[1]._event_callbacks:
                cb(PiEvent(type="error", content="timeout"))

        task = asyncio.create_task(simulate())
        events = await collect(service, 1, "hola")
        await task

        assert service.get_state(1) is PiSessionState.PI_ACTIVO


class TestIsolation:
    async def test_users_have_independent_sessions(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        await enter_pi(service, 2)
        assert 1 in agents
        assert 2 in agents
        assert agents[1] is not agents[2]

    async def test_user_not_in_pi_mode(self) -> None:
        service, agents = make_service()
        assert service.get_state(1) is PiSessionState.NORMAL

class TestModelCommand:
    async def test_model_lists_current_and_available(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/model")
        text = " ".join(e.content for e in events)
        assert "anthropic/claude-x" in text
        assert "openai/gpt-5" in text
        assert "Modelo activo" in text

    async def test_model_without_active_model(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].current_model = None
        events = await collect(service, 1, "/model")
        text = " ".join(e.content for e in events)
        assert "ninguno" in text.lower()

    async def test_model_set_unique_match(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/model GPT")
        assert agents[1].set_model_calls == [("openai", "gpt-5")]
        assert any("Modelo cambiado" in e.content for e in events)

    async def test_model_exact_provider_id(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/model anthropic/claude-x")
        assert agents[1].set_model_calls == [("anthropic", "claude-x")]
        assert any("Modelo cambiado" in e.content for e in events)

    async def test_model_multiple_matches_lists_candidates(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].models = [
            {"id": "m1", "name": "Modelo Uno", "provider": "p"},
            {"id": "m2", "name": "Modelo Dos", "provider": "p"},
        ]
        events = await collect(service, 1, "/model modelo")
        assert agents[1].set_model_calls == []
        text = " ".join(e.content for e in events)
        assert "p/m1" in text and "p/m2" in text

    async def test_model_no_match_shows_error(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "/model zzz")
        assert agents[1].set_model_calls == []
        assert any(e.type == "error" for e in events)

    async def test_model_pi_error_is_reported(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)

        async def fail(provider: str, model_id: str) -> dict:
            raise PiModeError("rpc_error", user_message="El agente no pudo completar el cambio")

        agents[1].set_model = fail  # type: ignore[method-assign]
        events = await collect(service, 1, "/model GPT")
        assert any("El agente no pudo completar el cambio" in e.content for e in events)


class TestForwardCommands:
    async def test_sh_runs_bash(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "//sh ls -la")
        assert agents[1].bash_calls == ["ls -la"]
        text = " ".join(e.content for e in events)
        assert "total 0" in text
        assert "0" in text

    async def test_sh_without_command_shows_usage(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "//sh")
        assert agents[1].bash_calls == []
        assert any("Uso" in e.content for e in events)

    async def test_sh_truncated_output_is_flagged(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].bash_result = {
            "output": "parcial",
            "exitCode": 0,
            "cancelled": False,
            "truncated": True,
        }
        events = await collect(service, 1, "//sh cat big.txt")
        assert any("truncad" in e.content for e in events)

    async def test_slash_command_is_forwarded_as_prompt(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        await collect(service, 1, "//fix-tests")
        assert agents[1].prompts == ["/fix-tests"]

    async def test_slash_command_with_args(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        await collect(service, 1, "//fix-tests src/")
        assert agents[1].prompts == ["/fix-tests src/"]

    async def test_unknown_slash_command_lists_available(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        events = await collect(service, 1, "//otro")
        assert agents[1].prompts == []
        text = " ".join(e.content for e in events)
        assert "fix-tests" in text

    async def test_sh_is_reserved_over_discoverable_commands(self) -> None:
        service, agents = make_service()
        await enter_pi(service, 1)
        agents[1].commands = [{"name": "sh", "description": "trampa", "source": "extension"}]
        await collect(service, 1, "//sh echo hola")
        assert agents[1].bash_calls == ["echo hola"]
        assert agents[1].prompts == []

    async def test_outside_pi_mode_shows_hint(self) -> None:
        service, agents = make_service()
        events = await collect(service, 1, "/model")
        assert any("/agent" in e.content for e in events)
        events = await collect(service, 1, "//sh ls")
        assert any("/agent" in e.content for e in events)
        assert 1 not in agents


class TestVoiceMessages:
    @staticmethod
    def attachment() -> IncomingAttachment:
        return IncomingAttachment(
            file_id="voice-id",
            file_name=None,
            mime_type="audio/ogg",
            file_size=9,
            kind="voice",
        )

    async def test_voice_transcription_is_sent_as_prompt_with_caption(self) -> None:
        transcriber = FakeAudioTranscriber()
        service, agents = make_service(transcriber)
        await enter_pi(service, 1)

        events = [
            event
            async for event in service.handle_voice(
                1, self.attachment(), b"OggSvoice", caption="Resume esto",
            )
        ]

        assert transcriber.calls == [(b"OggSvoice", "audio/ogg")]
        assert len(agents[1].prompts) == 1
        assert "Resume esto" in agents[1].prompts[0]
        assert "Hola, ¿puedes ayudarme?" in agents[1].prompts[0]
        assert agents[1].prompt_images == [None]
        assert any(event.type == "text_final" for event in events)

    async def test_voice_transcription_is_steered_while_pi_is_working(self) -> None:
        transcriber = FakeAudioTranscriber()
        service, agents = make_service(transcriber)
        await enter_pi(service, 1)
        agents[1]._is_streaming = True

        await anext(service.handle_voice(
            1, self.attachment(), b"OggSvoice", caption=None,
        ).__aiter__())

        assert agents[1].prompts == []
        assert len(agents[1].steers) == 1
        assert "Hola, ¿puedes ayudarme?" in agents[1].steers[0]

    async def test_voice_transcription_failure_never_reaches_pi(self) -> None:
        transcriber = FakeAudioTranscriber(
            error=AudioTranscriptionError("transcription_failed"),
        )
        service, agents = make_service(transcriber)
        await enter_pi(service, 1)

        events = [
            event
            async for event in service.handle_voice(
                1, self.attachment(), b"OggSvoice", caption=None,
            )
        ]

        assert agents[1].prompts == []
        assert agents[1].steers == []
        assert any(
            event.type == "error" and event.content.startswith(PI_PREFIX)
            for event in events
        )
        assert all("transcription_failed" not in event.content for event in events)

    async def test_voice_is_not_transcribed_before_pi_is_active(self) -> None:
        transcriber = FakeAudioTranscriber()
        service, agents = make_service(transcriber)

        events = [
            event
            async for event in service.handle_voice(
                1, self.attachment(), b"OggSvoice", caption="1",
            )
        ]

        assert transcriber.calls == []
        assert agents == {}
        assert events

    async def test_voice_is_not_processed_during_selection_or_exit_confirmation(self) -> None:
        selection_transcriber = FakeAudioTranscriber()
        selection_service, _ = make_service(selection_transcriber)
        await collect(selection_service, 1, "/agent")
        selection_events = [
            event
            async for event in selection_service.handle_voice(
                1, self.attachment(), b"OggSvoice", caption="1",
            )
        ]

        confirm_transcriber = FakeAudioTranscriber()
        confirm_service, confirm_agents = make_service(confirm_transcriber)
        await enter_pi(confirm_service, 2)
        confirm_agents[2]._is_streaming = True
        await collect(confirm_service, 2, "/exitagent")
        confirm_events = [
            event
            async for event in confirm_service.handle_voice(
                2, self.attachment(), b"OggSvoice", caption="sí",
            )
        ]

        assert selection_service.get_state(1) is PiSessionState.SELECCION_PROYECTO
        assert confirm_service.get_state(2) is PiSessionState.CONFIRMAR_SALIDA
        assert selection_transcriber.calls == []
        assert confirm_transcriber.calls == []
        assert selection_events and confirm_events

    async def test_transcribed_slash_text_is_not_a_telegram_command(self) -> None:
        transcriber = FakeAudioTranscriber(transcript="/exitagent")
        service, agents = make_service(transcriber)
        await enter_pi(service, 1)

        await anext(service.handle_voice(
            1, self.attachment(), b"OggSvoice", caption=None,
        ).__aiter__())

        assert service.get_state(1) is PiSessionState.PI_ACTIVO
        assert len(agents[1].prompts) == 1
        assert "\n/exitagent\n" in agents[1].prompts[0]
