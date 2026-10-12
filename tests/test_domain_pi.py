"""Tests del dominio Pi: tipos, eventos y puerto."""

from __future__ import annotations

import pytest

from oraculo.domain.pi import (
    PiAgentPort,
    PiEvent,
    PiModeError,
    PiSessionState,
    PiToolEvent,
)


class TestPiSessionState:
    def test_states_exist(self) -> None:
        assert PiSessionState.NORMAL.value == "normal"
        assert PiSessionState.PI_ACTIVO.value == "pi_activo"
        assert PiSessionState.CONFIRMAR_SALIDA.value == "confirmar_salida"
        assert PiSessionState.PI_ERROR.value == "pi_error"


class TestPiEvent:
    def test_text_delta_event(self) -> None:
        event = PiEvent(type="text_delta", content="hola")
        assert event.type == "text_delta"
        assert event.content == "hola"
        assert event.tool_name is None

    def test_message_end_event(self) -> None:
        event = PiEvent(type="message_end", content="respuesta completa")
        assert event.type == "message_end"
        assert event.content == "respuesta completa"

    def test_error_event(self) -> None:
        event = PiEvent(type="error", content="modelo no disponible")
        assert event.type == "error"
        assert event.content == "modelo no disponible"

    def test_event_is_frozen(self) -> None:
        event = PiEvent(type="text_delta", content="hola")
        with pytest.raises(AttributeError):
            event.content = "otro"  # type: ignore[misc]


class TestPiToolEvent:
    def test_tool_start(self) -> None:
        event = PiToolEvent(type="tool_start", tool_name="read")
        assert event.type == "tool_start"
        assert event.tool_name == "read"

    def test_tool_end(self) -> None:
        event = PiToolEvent(type="tool_end", tool_name="bash")
        assert event.type == "tool_end"
        assert event.tool_name == "bash"

    def test_tool_event_is_frozen(self) -> None:
        event = PiToolEvent(type="tool_start", tool_name="read")
        with pytest.raises(AttributeError):
            event.tool_name = "otro"  # type: ignore[misc]


class TestPiModeError:
    def test_error_has_reason(self) -> None:
        error = PiModeError("process_crashed")
        assert error.reason == "process_crashed"

    def test_error_message(self) -> None:
        error = PiModeError("model_error")
        assert "model_error" in str(error)

    def test_error_with_user_message(self) -> None:
        error = PiModeError("timeout", user_message="Pi tardó demasiado")
        assert error.reason == "timeout"
        assert error.user_message == "Pi tardó demasiado"

    def test_error_without_user_message(self) -> None:
        error = PiModeError("process_crashed")
        assert error.user_message is None


class TestPiAgentPort:
    def test_port_is_protocol(self) -> None:
        assert hasattr(PiAgentPort, "send_prompt")
        assert hasattr(PiAgentPort, "send_steer")
        assert hasattr(PiAgentPort, "abort")
        assert hasattr(PiAgentPort, "new_session")
        assert hasattr(PiAgentPort, "get_state")
        assert hasattr(PiAgentPort, "set_model")
        assert hasattr(PiAgentPort, "get_available_models")
        assert hasattr(PiAgentPort, "get_commands")
        assert hasattr(PiAgentPort, "bash")
        assert hasattr(PiAgentPort, "on_event")
        assert hasattr(PiAgentPort, "on_tool_event")
        assert hasattr(PiAgentPort, "close")


class TestProjectSelectionDomain:
    def test_selection_states_exist(self) -> None:
        assert PiSessionState.SELECCION_PROYECTO.value == "seleccion_proyecto"
        assert PiSessionState.NOMBRE_PROYECTO.value == "nombre_proyecto"

    def test_pi_project_is_frozen(self) -> None:
        from oraculo.domain.pi import PiProject

        project = PiProject(name="demo", path="/tmp/raiz/demo")
        assert project.name == "demo"
        assert project.path == "/tmp/raiz/demo"
        with pytest.raises(AttributeError):
            project.name = "otro"  # type: ignore[misc]

    def test_project_store_port_is_protocol(self) -> None:
        from oraculo.domain.pi import PiProjectStorePort

        assert hasattr(PiProjectStorePort, "list_projects")
        assert hasattr(PiProjectStorePort, "resolve")
        assert hasattr(PiProjectStorePort, "create")
