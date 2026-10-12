"""Tests del flujo de selección/creación de proyecto al entrar en modo Pi."""

from __future__ import annotations

from collections.abc import Callable

from fakes import FakeProjectStore

from oraculo.application.pi_mode import (
    ROOT_PROJECT_LABEL,
    PiDisplayEvent,
    PiModeService,
)
from oraculo.domain.pi import PiEvent, PiModeError, PiProject, PiSessionState


class StubAgent:
    """Agente mínimo: solo registra prompts y cierre."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.closed = False
        self._event_callbacks: list[Callable[[PiEvent], None]] = []

    async def send_prompt(self, text: str, images: list[bytes] | None = None) -> None:
        self.prompts.append(text)
        # Cierra el turno como Pi real para que `_handle_message` termine.
        for callback in self._event_callbacks:
            callback(PiEvent(type="message_end", content="ok"))
        for callback in self._event_callbacks:
            callback(PiEvent(type="settled", content=""))

    async def get_state(self) -> dict[str, object]:
        return {"isStreaming": False}

    def on_event(self, callback: Callable[[PiEvent], None]) -> None:
        self._event_callbacks.append(callback)

    def on_tool_event(self, callback: object) -> None:
        pass

    async def close(self) -> None:
        self.closed = True


class Harness:
    def __init__(
        self,
        store: FakeProjectStore | None = None,
        factory_error: PiModeError | None = None,
        *,
        root_project: PiProject | None = None,
        root_user_ids: frozenset[int] = frozenset(),
    ) -> None:
        self.store = store if store is not None else FakeProjectStore(("alfa", "beta"))
        self.factory_error = factory_error
        self.launches: list[tuple[int, str]] = []
        self.agents: list[StubAgent] = []

        async def factory(user_id: int, cwd: str) -> StubAgent:
            self.launches.append((user_id, cwd))
            if self.factory_error is not None:
                raise self.factory_error
            agent = StubAgent()
            self.agents.append(agent)
            return agent

        self.service = PiModeService(  # type: ignore[arg-type]
            agent_factory=factory,
            project_store=self.store,
            root_project=root_project,
            root_user_ids=root_user_ids,
        )

    async def send(self, text: str, user_id: int = 1) -> list[PiDisplayEvent]:
        return [e async for e in self.service.handle(user_id, text)]

    def state(self, user_id: int = 1) -> PiSessionState:
        return self.service.get_state(user_id)


def text_of(events: list[PiDisplayEvent]) -> str:
    return "\n".join(e.content for e in events)


ROOT = FakeProjectStore.ROOT


class TestListing:
    async def test_pi_lists_projects_without_launching(self) -> None:
        h = Harness()
        events = await h.send("/agent")
        body = text_of(events)
        assert h.launches == []
        assert h.state() is PiSessionState.SELECCION_PROYECTO
        assert body.startswith("🤖💻")
        assert (
            "Elige el proyecto en el que trabajarás en modo agente "
            "(responde con el número):"
        ) in body
        assert "trabajará Pi" not in body
        assert "1. alfa" in body
        assert "2. beta" in body
        assert "3. Crear un proyecto nuevo" in body

    async def test_listing_does_not_expose_paths(self) -> None:
        h = Harness()
        body = text_of(await h.send("/agent"))
        assert ROOT not in body

    async def test_empty_root_offers_only_create(self) -> None:
        h = Harness(FakeProjectStore(()))
        body = text_of(await h.send("/agent"))
        assert "No hay proyectos" in body
        assert "1. Crear un proyecto nuevo" in body
        assert h.state() is PiSessionState.SELECCION_PROYECTO

    async def test_pi_during_selection_relists_without_launching(self) -> None:
        h = Harness()
        await h.send("/agent")
        body = text_of(await h.send("/agent"))
        assert "1. alfa" in body
        assert h.launches == []
        assert h.state() is PiSessionState.SELECCION_PROYECTO


class TestSelection:
    async def test_number_launches_pi_in_project(self) -> None:
        h = Harness()
        await h.send("/agent")
        body = text_of(await h.send("2"))
        assert h.launches == [(1, f"{ROOT}/beta")]
        assert h.state() is PiSessionState.PI_ACTIVO
        assert "beta" in body and body.startswith("🤖💻")

    async def test_number_with_spaces_is_accepted(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("  1 ")
        assert h.launches == [(1, f"{ROOT}/alfa")]

    async def test_messages_after_selection_go_to_pi(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("1")
        await h.send("hola")
        assert h.agents[0].prompts == ["hola"]

    async def test_invalid_choices_stay_in_selection(self) -> None:
        h = Harness()
        await h.send("/agent")
        for bad in ("0", "4", "-1", "abc", "alfa", "1.5"):
            events = await h.send(bad)
            assert any(e.type == "error" and e.content.startswith("🤖💻") for e in events), bad
            assert "1. alfa" in text_of(events), bad
            assert h.state() is PiSessionState.SELECCION_PROYECTO
        assert h.launches == []

    async def test_ordinary_message_not_sent_to_pi(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("arregla los tests")
        assert h.launches == []
        assert h.agents == []

    async def test_exitagent_cancels_selection(self) -> None:
        h = Harness()
        await h.send("/agent")
        body = text_of(await h.send("/exitagent"))
        assert h.state() is PiSessionState.NORMAL
        assert h.launches == []
        assert body.startswith("🤖💻")

    async def test_project_vanished_after_listing(self) -> None:
        h = Harness()
        await h.send("/agent")
        h.store.names = ["beta"]
        events = await h.send("1")
        assert any(e.type == "error" for e in events)
        assert h.state() is PiSessionState.SELECCION_PROYECTO
        assert h.launches == []

    async def test_selection_is_per_user(self) -> None:
        h = Harness()
        await h.send("/agent", user_id=1)
        events = await h.send("1", user_id=2)
        assert events == []
        assert h.state(2) is PiSessionState.NORMAL
        assert h.state(1) is PiSessionState.SELECCION_PROYECTO


class TestCreate:
    async def test_create_option_asks_for_name(self) -> None:
        h = Harness()
        await h.send("/agent")
        body = text_of(await h.send("3"))
        assert h.state() is PiSessionState.NOMBRE_PROYECTO
        assert "nombre" in body.lower()
        assert h.launches == []

    async def test_valid_name_creates_and_launches(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("3")
        body = text_of(await h.send("nuevo"))
        assert h.store.created == ["nuevo"]
        assert h.launches == [(1, f"{ROOT}/nuevo")]
        assert h.state() is PiSessionState.PI_ACTIVO
        assert "nuevo" in body

    async def test_existing_name_reuses_with_notice(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("3")
        body = text_of(await h.send("alfa"))
        assert h.store.created == []
        assert h.launches == [(1, f"{ROOT}/alfa")]
        assert "ya existía" in body

    async def test_invalid_name_asks_again(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("3")
        events = await h.send("../fuera")
        assert any(e.type == "error" and e.content.startswith("🤖💻") for e in events)
        assert h.state() is PiSessionState.NOMBRE_PROYECTO
        assert h.launches == []
        assert h.store.created == []

    async def test_create_from_empty_root(self) -> None:
        h = Harness(FakeProjectStore(()))
        await h.send("/agent")
        await h.send("1")
        await h.send("primero")
        assert h.launches == [(1, f"{ROOT}/primero")]

    async def test_exitagent_cancels_naming(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("3")
        await h.send("/exitagent")
        assert h.state() is PiSessionState.NORMAL
        assert h.launches == []

    async def test_pi_during_naming_relists(self) -> None:
        h = Harness()
        await h.send("/agent")
        await h.send("3")
        body = text_of(await h.send("/agent"))
        assert h.state() is PiSessionState.SELECCION_PROYECTO
        assert "1. alfa" in body


class TestErrors:
    async def test_list_failure_returns_to_normal(self) -> None:
        store = FakeProjectStore(
            list_error=PiModeError("projects_unavailable", user_message="No se pudo leer la carpeta de proyectos.")
        )
        h = Harness(store)
        events = await h.send("/agent")
        assert any(e.type == "error" and "No se pudo leer" in e.content for e in events)
        assert h.state() is PiSessionState.NORMAL
        assert h.launches == []

    async def test_create_failure_keeps_asking_name(self) -> None:
        store = FakeProjectStore(
            ("alfa",),
            create_error=PiModeError("project_create_failed", user_message="No se pudo crear la carpeta del proyecto."),
        )
        h = Harness(store)
        await h.send("/agent")
        await h.send("2")
        events = await h.send("nuevo")
        assert any(e.type == "error" and "No se pudo crear" in e.content for e in events)
        assert h.state() is PiSessionState.NOMBRE_PROYECTO
        assert h.launches == []

    async def test_launch_failure_allows_retry(self) -> None:
        h = Harness(
            factory_error=PiModeError(
                "sandbox_unavailable",
                user_message=(
                    "No se encontró el ejecutable necesario para el modo agente "
                    "o no está en el PATH."
                ),
            )
        )
        await h.send("/agent")
        events = await h.send("1")
        assert any(e.type == "error" and "ejecutable necesario para el modo agente" in e.content for e in events)
        assert h.state() is PiSessionState.SELECCION_PROYECTO
        h.factory_error = None
        await h.send("1")
        assert h.state() is PiSessionState.PI_ACTIVO
        assert len(h.launches) == 2


class TestRoutingContract:
    async def test_selection_states_count_as_pi_mode(self) -> None:
        """El handler solo enruta texto a Pi si `is_in_pi_mode`; la respuesta «1» no debe ir al LLM."""
        h = Harness()
        assert h.service.is_in_pi_mode(1) is False
        await h.send("/agent")
        assert h.service.is_in_pi_mode(1) is True
        await h.send("3")
        assert h.service.is_in_pi_mode(1) is True
        await h.send("/exitagent")
        assert h.service.is_in_pi_mode(1) is False


ROOT_PATH = "/repo/oraculo"


def _root_harness(**kwargs: object) -> Harness:
    return Harness(
        root_project=PiProject(name=ROOT_PROJECT_LABEL, path=ROOT_PATH),
        root_user_ids=frozenset({1}),
        **kwargs,  # type: ignore[arg-type]
    )


class TestRootOption:
    async def test_root_user_sees_root_option(self) -> None:
        h = _root_harness()
        body = text_of(await h.send("/agent", user_id=1))
        # alfa=1, beta=2, crear=3, raíz=4
        assert f"4. {ROOT_PROJECT_LABEL}" in body

    async def test_non_root_user_does_not_see_root_option(self) -> None:
        h = _root_harness()
        body = text_of(await h.send("/agent", user_id=2))
        assert ROOT_PROJECT_LABEL not in body
        assert "3. Crear un proyecto nuevo" in body

    async def test_no_root_project_hides_option_for_everyone(self) -> None:
        h = Harness()  # sin root_project
        body = text_of(await h.send("/agent", user_id=1))
        assert ROOT_PROJECT_LABEL not in body

    async def test_root_user_selects_root_launches_at_repo_root(self, caplog) -> None:
        import logging

        h = _root_harness()
        await h.send("/agent", user_id=1)
        with caplog.at_level(logging.INFO):
            body = text_of(await h.send("4", user_id=1))
        assert h.launches == [(1, ROOT_PATH)]
        assert h.state(1) is PiSessionState.PI_ACTIVO
        assert body.startswith("🤖💻")
        assert "raíz del proyecto Oraculo" in body
        assert ROOT_PATH not in body
        assert any("raíz del proyecto" in r.message and "1" in r.message for r in caplog.records)

    async def test_non_root_selecting_root_index_is_invalid(self) -> None:
        h = _root_harness()
        await h.send("/agent", user_id=2)
        body = text_of(await h.send("4", user_id=2))
        assert h.launches == []
        assert h.state(2) is PiSessionState.SELECCION_PROYECTO
        assert "no válida" in body.lower()

    async def test_root_user_out_of_range_is_invalid(self) -> None:
        h = _root_harness()
        await h.send("/agent", user_id=1)
        body = text_of(await h.send("5", user_id=1))
        assert h.launches == []
        assert "del 1 al 4" in body

    async def test_root_user_still_selects_existing_project(self) -> None:
        h = _root_harness()
        await h.send("/agent", user_id=1)
        body = text_of(await h.send("1", user_id=1))
        assert h.launches == [(1, f"{ROOT}/alfa")]
        assert "alfa" in body

    async def test_root_user_still_creates_project(self) -> None:
        h = _root_harness()
        await h.send("/agent", user_id=1)
        await h.send("3", user_id=1)  # crear nuevo
        body = text_of(await h.send("gamma", user_id=1))
        assert h.launches == [(1, f"{ROOT}/gamma")]
        assert "gamma" in body
