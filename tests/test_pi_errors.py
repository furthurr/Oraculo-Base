"""Tests de errores del modo Pi: proceso caído, timeout, Docker no disponible."""

from __future__ import annotations

import asyncio

import pytest

from oraculo.domain.pi import PiEvent, PiModeError, PiSessionState, PiToolEvent
from fakes import FakeProjectStore, enter_pi

from oraculo.application.pi_mode import PiDisplayEvent, PiModeService


class FailingPiAgent:
    """Agente Pi simulado que falla al recibir prompts."""

    def __init__(self, fail_mode: str = "prompt") -> None:
        self.fail_mode = fail_mode
        self._event_callbacks: list = []
        self._tool_callbacks: list = []
        self.closed = False

    async def send_prompt(self, text, images=None):
        if self.fail_mode == "prompt":
            raise PiModeError("model_error", user_message="El modelo no responde")
        if self.fail_mode == "crash":
            raise PiModeError("process_crashed", user_message="El proceso del agente se detuvo")

    async def send_steer(self, text, images=None):
        raise PiModeError("process_crashed")

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


def make_failing_service(fail_mode: str = "prompt"):
    agent = FailingPiAgent(fail_mode)

    async def factory(user_id: int, cwd: str):
        return agent

    return PiModeService(agent_factory=factory, project_store=FakeProjectStore()), agent


class TestModelError:
    async def test_model_error_shows_message(self) -> None:
        service, agent = make_failing_service("prompt")
        await enter_pi(service, 1)
        events = await collect(service, 1, "hola")
        errors = [e for e in events if e.type == "error"]
        assert len(errors) == 1
        assert "🤖💻" in errors[0].content
        assert "modelo" in errors[0].content.lower() or "responde" in errors[0].content.lower()

    async def test_model_error_stays_in_pi_mode(self) -> None:
        service, agent = make_failing_service("prompt")
        await enter_pi(service, 1)
        await collect(service, 1, "hola")
        assert service.get_state(1) is PiSessionState.PI_ACTIVO


class TestProcessCrash:
    async def test_crash_shows_error(self) -> None:
        service, agent = make_failing_service("crash")
        await enter_pi(service, 1)
        events = await collect(service, 1, "hola")
        errors = [e for e in events if e.type == "error"]
        assert len(errors) == 1
        assert "🤖💻" in errors[0].content

    async def test_crash_stays_in_pi_mode(self) -> None:
        service, agent = make_failing_service("crash")
        await enter_pi(service, 1)
        await collect(service, 1, "hola")
        assert service.get_state(1) is PiSessionState.PI_ACTIVO

    async def test_can_retry_after_crash(self) -> None:
        service, agent = make_failing_service("crash")
        await enter_pi(service, 1)
        await collect(service, 1, "hola")
        events = await collect(service, 1, "otro intento")
        assert service.get_state(1) is PiSessionState.PI_ACTIVO


async def collect(service, user_id, text):
    return [e async for e in service.handle(user_id, text)]