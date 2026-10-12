"""Regresión del turno de Pi: aislamiento, fin por `settled`, timeout y relanzamiento.

Spec: `.sdd/specs/modo-pi/bugfix-lector-rpc/` (Req 2.1, 2.3–2.4, 3.x, 4.x).
Cada test va envuelto en `wait_for`: un cuelgue es un fallo, no un test infinito.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest

from fakes import FakeProjectStore, enter_pi

from oraculo.application.pi_mode import PiDisplayEvent, PiModeService
from oraculo.domain.pi import PiEvent, PiModeError, PiProject, PiSessionState, PiToolEvent

SESSION_FILE = "/Users/alguien/.pi/agent/sessions/proyecto/2026-09-30-secreto.jsonl"
TEST_TIMEOUT_S = 5.0


class ScriptedAgent:
    """Agente fake: cada prompt ejecuta un guion de eventos definido por el test."""

    def __init__(self, script: Callable[[ScriptedAgent], object] | None = None) -> None:
        self.script = script
        self.prompts: list[str] = []
        self.aborted = False
        self.closed = False
        self.session_file: str | None = SESSION_FILE
        self.state_error: PiModeError | None = None
        self._event_callbacks: list[Callable[[PiEvent], None]] = []
        self._tool_callbacks: list[Callable[[PiToolEvent], None]] = []

    # --- puerto -----------------------------------------------------------
    async def send_prompt(self, text: str, images: list[bytes] | None = None) -> None:
        self.prompts.append(text)
        if self.script is not None:
            asyncio.ensure_future(self._run_script())

    async def _run_script(self) -> None:
        await asyncio.sleep(0.01)
        assert self.script is not None
        result = self.script(self)
        if asyncio.iscoroutine(result):
            await result

    async def send_steer(
        self, text: str, images: list[bytes] | None = None
    ) -> None:
        return None

    async def abort(self) -> None:
        self.aborted = True

    async def new_session(self) -> None:
        return None

    async def get_state(self) -> dict[str, object]:
        if self.state_error is not None:
            raise self.state_error
        state: dict[str, object] = {"isStreaming": False}
        if self.session_file is not None:
            state["sessionFile"] = self.session_file
        return state

    def on_event(self, callback: Callable[[PiEvent], None]) -> None:
        self._event_callbacks.append(callback)

    def on_tool_event(self, callback: Callable[[PiToolEvent], None]) -> None:
        self._tool_callbacks.append(callback)

    async def close(self) -> None:
        self.closed = True

    # --- helpers de emisión -----------------------------------------------
    def event(self, type_: str, content: str = "") -> None:
        for cb in list(self._event_callbacks):
            cb(PiEvent(type=type_, content=content))

    def tool(self, type_: str, name: str) -> None:
        for cb in list(self._tool_callbacks):
            cb(PiToolEvent(type=type_, tool_name=name))


def answer(text: str) -> Callable[[ScriptedAgent], None]:
    def script(agent: ScriptedAgent) -> None:
        agent.tool("tool_start", "read")
        agent.event("message_end", text)
        agent.event("settled")

    return script


class Harness:
    def __init__(
        self,
        *,
        idle_timeout_s: float = 2.0,
        root_project: PiProject | None = None,
        root_user_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.agents: list[ScriptedAgent] = []
        self.launches: list[tuple[str, str | None]] = []
        self.fail_launches: list[bool] = []
        self.dead_on_arrival: list[bool] = []
        self.next_script: Callable[[ScriptedAgent], object] | None = answer("ok")

        async def factory(
            user_id: int, cwd: str, session_file: str | None = None
        ) -> ScriptedAgent:
            self.launches.append((cwd, session_file))
            if self.fail_launches and self.fail_launches.pop(0):
                raise PiModeError("process_crashed", user_message="El agente no arrancó")
            agent = ScriptedAgent(self.next_script)
            if self.dead_on_arrival and self.dead_on_arrival.pop(0):
                agent.state_error = PiModeError("process_crashed")
            self.agents.append(agent)
            return agent

        self.service = PiModeService(  # type: ignore[arg-type]
            agent_factory=factory,
            project_store=FakeProjectStore(("alfa",)),
            root_project=root_project,
            root_user_ids=root_user_ids,
            idle_timeout_s=idle_timeout_s,
        )

    async def send(self, text: str, user_id: int = 1) -> list[PiDisplayEvent]:
        async def run() -> list[PiDisplayEvent]:
            return [e async for e in self.service.handle(user_id, text)]

        return await asyncio.wait_for(run(), timeout=TEST_TIMEOUT_S)

    @property
    def agent(self) -> ScriptedAgent:
        return self.agents[-1]


def of_type(events: list[PiDisplayEvent], type_: str) -> list[str]:
    return [e.content for e in events if e.type == type_]


def all_text(events: list[PiDisplayEvent]) -> str:
    return "\n".join(e.content for e in events)


class TestIsolation:
    async def test_consecutive_messages_emit_each_notice_once(self) -> None:
        """Req 4.2 — antes se acumulaban callbacks: 1, 2, 3 avisos."""
        h = Harness()
        await enter_pi(h.service, 1)
        counts = []
        for i in range(3):
            events = await h.send(f"mensaje {i}")
            counts.append(len(of_type(events, "tool_notice")))
            assert of_type(events, "text_final") == ["ok"]
        assert counts == [1, 1, 1]

    async def test_callbacks_are_registered_once_per_agent(self) -> None:
        """Req 4.1 (C5) — antes se añadía un par de callbacks por mensaje (fuga)."""
        h = Harness()
        await enter_pi(h.service, 1)
        for i in range(3):
            await h.send(f"mensaje {i}")
        assert len(h.agent._event_callbacks) == 1
        assert len(h.agent._tool_callbacks) == 1

    async def test_event_after_turn_end_is_dropped_safely(self) -> None:
        """Req 4.1."""
        h = Harness()
        await enter_pi(h.service, 1)
        await h.send("hola")
        h.agent.event("text_delta", "tarde")
        h.agent.tool("tool_start", "bash")
        h.agent.script = answer("siguiente")
        events = await h.send("otra")
        assert "tarde" not in all_text(events)
        assert len(of_type(events, "tool_notice")) == 1


class TestTurnEnd:
    async def test_all_assistant_messages_until_settled_are_delivered(self) -> None:
        """Req 3.1, 3.2 — antes el turno terminaba en el primer `message_end`."""

        async def script(agent: ScriptedAgent) -> None:
            agent.event("message_end", "primero reviso")
            agent.tool("tool_start", "bash")
            await asyncio.sleep(0.02)
            agent.tool("tool_update", "bash")
            agent.tool("tool_end", "bash")
            agent.event("message_end", "encontré la config")
            agent.event("settled")

        h = Harness()
        h.next_script = script
        await enter_pi(h.service, 1)
        events = await h.send("revisa el STT")
        assert of_type(events, "text_final") == ["primero reviso", "encontré la config"]
        assert len(of_type(events, "tool_notice")) == 1

    async def test_idle_timeout_aborts_and_reports(self) -> None:
        """Req 3.3 — antes esperaba indefinidamente."""
        h = Harness(idle_timeout_s=0.1)
        h.next_script = None
        await enter_pi(h.service, 1)
        events = await h.send("haz algo")
        errors = of_type(events, "error")
        assert len(errors) == 1 and errors[0].startswith("🤖💻")
        assert "no respondió" in errors[0]
        assert h.agent.aborted
        assert h.service.get_state(1) is PiSessionState.PI_ACTIVO

    async def test_activity_resets_idle_timer(self) -> None:
        """Req 3.3 — el timeout es por inactividad, no por duración total."""

        async def script(agent: ScriptedAgent) -> None:
            for _ in range(6):
                await asyncio.sleep(0.05)
                agent.event("activity")
                agent.tool("tool_update", "bash")
            agent.event("message_end", "listo")
            agent.event("settled")

        h = Harness(idle_timeout_s=0.12)
        h.next_script = script
        await enter_pi(h.service, 1)
        events = await h.send("tarea larga")
        assert of_type(events, "error") == []
        assert of_type(events, "text_final") == ["listo"]
        assert not h.agent.aborted

    async def test_model_error_still_ends_turn(self) -> None:
        """Req 3.4 — comportamiento a preservar."""

        def script(agent: ScriptedAgent) -> None:
            agent.event("error", "model_error: fallo")

        h = Harness()
        h.next_script = script
        await enter_pi(h.service, 1)
        events = await h.send("hola")
        assert of_type(events, "error") == ["🤖💻 model_error: fallo"]

    async def test_closed_during_turn_reports_and_ends(self) -> None:
        """Req 2.1 — caída del lector durante el turno."""

        def script(agent: ScriptedAgent) -> None:
            agent.tool("tool_start", "bash")
            agent.event("closed")

        h = Harness()
        h.next_script = script
        await enter_pi(h.service, 1)
        events = await h.send("revisa")
        errors = of_type(events, "error")
        assert len(errors) == 1 and "se cerró" in errors[0]
        assert h.service.is_in_pi_mode(1)


class TestRelaunch:
    async def _crash(self, h: Harness) -> None:
        def script(agent: ScriptedAgent) -> None:
            agent.event("closed")

        h.agent.script = script
        await h.send("primera")

    async def test_resumes_previous_session(self) -> None:
        """Req 2.3."""
        h = Harness()
        await enter_pi(h.service, 1)
        await self._crash(h)
        h.next_script = answer("sigo donde iba")
        events = await h.send("¿qué encontraste?")
        assert h.launches[-1][1] == SESSION_FILE
        assert "reanud" in all_text(events)
        assert of_type(events, "text_final") == ["sigo donde iba"]
        assert h.agent.prompts == ["¿qué encontraste?"]
        assert h.agents[0].closed

    async def test_falls_back_to_clean_session(self) -> None:
        """Req 2.3.1 — la reanudación falla y se abre una sesión limpia."""
        h = Harness()
        await enter_pi(h.service, 1)
        await self._crash(h)
        h.fail_launches = [True, False]
        events = await h.send("¿qué encontraste?")
        assert [s for _, s in h.launches[-2:]] == [SESSION_FILE, None]
        assert "perdió" in all_text(events)
        assert of_type(events, "text_final") == ["ok"]

    async def test_resumed_agent_dead_on_arrival_falls_back(self) -> None:
        """Req 2.3.1 — Pi arranca con la sesión pero no responde a `get_state`."""
        h = Harness()
        await enter_pi(h.service, 1)
        await self._crash(h)
        h.dead_on_arrival = [True, False]
        events = await h.send("¿qué encontraste?")
        assert [s for _, s in h.launches[-2:]] == [SESSION_FILE, None]
        assert h.agents[-2].closed
        assert "perdió" in all_text(events)
        assert of_type(events, "text_final") == ["ok"]

    async def test_both_attempts_fail_keeps_pi_mode(self) -> None:
        """Req 2.3.2, 2.4."""
        h = Harness()
        await enter_pi(h.service, 1)
        await self._crash(h)
        h.fail_launches = [True, True]
        events = await h.send("¿qué encontraste?")
        assert "No se pudo reiniciar el agente" in all_text(events)
        assert h.service.get_state(1) is PiSessionState.PI_ERROR
        assert h.service.is_in_pi_mode(1)
        # Siguiente mensaje: vuelve a intentarlo y funciona.
        events = await h.send("reintento")
        assert of_type(events, "text_final") == ["ok"]
        assert h.service.get_state(1) is PiSessionState.PI_ACTIVO

    async def test_process_crashed_error_also_marks_crash(self) -> None:
        """Req 2.3 — `get_state` con `process_crashed` equivale a caída."""
        h = Harness()
        await enter_pi(h.service, 1)
        await h.send("hola")
        h.agent.state_error = PiModeError("process_crashed")
        events = await h.send("otra")
        assert of_type(events, "error")
        events = await h.send("y ahora")
        assert "reanud" in all_text(events)
        assert h.launches[-1][1] == SESSION_FILE

    async def test_session_file_never_shown_or_logged(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.3.3."""
        h = Harness()
        with caplog.at_level(logging.DEBUG):
            await enter_pi(h.service, 1)
            await self._crash(h)
            h.fail_launches = [True, False]
            events = await h.send("¿qué encontraste?")
        assert SESSION_FILE not in all_text(events)
        assert "secreto" not in caplog.text
        assert "/Users/alguien" not in caplog.text

    async def test_root_relaunch_keeps_label_and_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Req 2.3 + Req 4.1 de `modo-pi-root`: etiqueta sin ruta y log de raíz."""
        root = PiProject(name="Oraculo", path="/ruta/absoluta/oraculo")
        h = Harness(root_project=root, root_user_ids=frozenset({1}))
        await enter_pi(h.service, 1, choice="3")
        await self._crash(h)
        with caplog.at_level(logging.INFO, logger="oraculo.application.pi_mode"):
            events = await h.send("sigue")
        assert h.launches[-1][0] == "/ruta/absoluta/oraculo"
        assert "raíz del proyecto" in all_text(events)
        assert "/ruta/absoluta" not in all_text(events)
        assert "raíz del proyecto" in caplog.text

    async def test_exitagent_clears_crash_and_pi_starts_fresh(self) -> None:
        """D6 — `/agent` tras `/exitagent` no reanuda."""
        h = Harness()
        await enter_pi(h.service, 1)
        await self._crash(h)
        await h.send("/exitagent")
        await enter_pi(h.service, 1)
        await h.send("hola")
        assert h.launches[-1][1] is None
