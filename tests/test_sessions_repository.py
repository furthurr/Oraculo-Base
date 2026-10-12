"""Tests del repositorio de sesiones en memoria (Req 1.2, 3.6, 5.1, 5.4)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from oraculo.domain.models import ChatMessage, MessageRole, SessionKey
from oraculo.infrastructure.sessions import InMemorySessionRepository

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def user(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=text)


def assistant(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content=text)


async def test_append_crea_y_load_devuelve_la_sesion():
    repo = InMemorySessionRepository()
    key = SessionKey.private(1)

    await repo.append(key, user("hola"), assistant("qué tal"), now=NOW)
    state = await repo.load(key)

    assert state is not None
    assert [m.content for m in state.messages] == ["hola", "qué tal"]
    assert state.updated_at == NOW


async def test_load_sin_sesion_devuelve_none():
    repo = InMemorySessionRepository()

    assert await repo.load(SessionKey.private(1)) is None


async def test_append_conserva_el_orden_de_los_turnos():
    repo = InMemorySessionRepository()
    key = SessionKey.private(1)

    await repo.append(key, user("uno"), assistant("r1"), now=NOW)
    await repo.append(key, user("dos"), assistant("r2"), now=NOW)

    state = await repo.load(key)
    assert state is not None
    assert [m.content for m in state.messages] == ["uno", "r1", "dos", "r2"]


async def test_tope_por_sesion_descarta_los_turnos_mas_antiguos():
    repo = InMemorySessionRepository(max_stored_turns=2)
    key = SessionKey.private(1)

    await repo.append(key, user("uno"), assistant("r1"), now=NOW)
    await repo.append(key, user("dos"), assistant("r2"), now=NOW)
    await repo.append(key, user("tres"), assistant("r3"), now=NOW)

    state = await repo.load(key)
    assert state is not None
    assert [m.content for m in state.messages] == ["dos", "r2", "tres", "r3"]


async def test_tope_global_descarta_la_sesion_mas_antigua():
    repo = InMemorySessionRepository(max_sessions=2)
    primera, segunda, tercera = (
        SessionKey.private(1),
        SessionKey.private(2),
        SessionKey.private(3),
    )

    await repo.append(primera, user("a"), assistant("r"), now=NOW)
    await repo.append(segunda, user("b"), assistant("r"), now=NOW)
    await repo.append(tercera, user("c"), assistant("r"), now=NOW)

    assert await repo.load(primera) is None
    assert await repo.load(segunda) is not None
    assert await repo.load(tercera) is not None


async def test_load_actualiza_el_orden_lru():
    repo = InMemorySessionRepository(max_sessions=2)
    primera, segunda, tercera = (
        SessionKey.private(1),
        SessionKey.private(2),
        SessionKey.private(3),
    )

    await repo.append(primera, user("a"), assistant("r"), now=NOW)
    await repo.append(segunda, user("b"), assistant("r"), now=NOW)
    await repo.load(primera)  # la primera pasa a ser la más reciente
    await repo.append(tercera, user("c"), assistant("r"), now=NOW)

    assert await repo.load(primera) is not None
    assert await repo.load(segunda) is None


async def test_discard_elimina_la_sesion():
    repo = InMemorySessionRepository()
    key = SessionKey.group(-100)
    await repo.append(key, user("hola"), assistant("qué tal"), now=NOW)

    await repo.discard(key)

    assert await repo.load(key) is None


async def test_el_estado_previo_no_se_muta():
    repo = InMemorySessionRepository()
    key = SessionKey.private(1)
    await repo.append(key, user("uno"), assistant("r1"), now=NOW)
    anterior = await repo.load(key)
    assert anterior is not None

    await repo.append(key, user("dos"), assistant("r2"), now=NOW)

    assert [m.content for m in anterior.messages] == ["uno", "r1"]


async def test_compactacion_reemplaza_historial_y_conserva_actividad_y_clave():
    repo = InMemorySessionRepository()
    key = SessionKey.private(1)
    other = SessionKey.private(2)
    await repo.append(key, user("uno"), assistant("r1"), now=NOW)
    await repo.append(other, user("otro"), assistant("r2"), now=NOW)
    before = await repo.load(key)
    await repo.compact(key, "resumen")
    after = await repo.load(key)
    assert after is not None and before is not None
    assert after.summary == "resumen" and after.messages == ()
    assert after.updated_at == NOW
    assert before.messages == (user("uno"), assistant("r1"))
    assert (await repo.load(other)).summary is None
    await repo.append(key, user("dos"), assistant("r3"), now=NOW)
    assert (await repo.load(key)).summary == "resumen"
    await repo.discard(key)
    assert await repo.load(key) is None


async def test_lru_desaloja_resumen_junto_con_sesion():
    repo = InMemorySessionRepository(max_sessions=1)
    first, second = SessionKey.private(1), SessionKey.private(2)
    await repo.append(first, user("uno"), assistant("r1"), now=NOW)
    await repo.compact(first, "resumen acumulado")
    await repo.append(second, user("dos"), assistant("r2"), now=NOW)
    assert await repo.load(first) is None
    assert (await repo.load(second)).summary is None


async def test_la_creacion_se_registra_sin_contenido(caplog):
    repo = InMemorySessionRepository()
    key = SessionKey.private(1)

    with caplog.at_level(logging.DEBUG):
        await repo.append(key, user("secreto"), assistant("respuesta"), now=NOW)

    assert "Sesión creada" in caplog.text
    assert "private:1" in caplog.text
    assert "secreto" not in caplog.text
