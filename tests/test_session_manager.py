"""Tests del `SessionManager` (Req 1.4, 1.5, 2.2, 3.1–3.5, 5.3, 6.3)."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fakes import FakeClock, FakeSessionRepository

from oraculo.application.sessions import SessionManager
from oraculo.domain.models import (
    ChatMessage,
    MessageRole,
    SessionKey,
    SessionState,
)

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def user(text: str, author: str | None = None) -> ChatMessage:
    return ChatMessage(role=MessageRole.USER, content=text, author_name=author)


def assistant(text: str) -> ChatMessage:
    return ChatMessage(role=MessageRole.ASSISTANT, content=text)


def build_manager(*, max_turns: int = 15, idle_minutes: int = 240):
    repository = FakeSessionRepository()
    clock = FakeClock(NOW)
    manager = SessionManager(
        repository=repository,
        max_turns=max_turns,
        idle_minutes=idle_minutes,
        clock=clock,
    )
    return manager, repository, clock


async def test_ventana_vacia_sin_sesion():
    manager, _, _ = build_manager()

    assert await manager.load_window(SessionKey.private(1)) == ()


async def test_registro_y_ventana_completa():
    manager, _, _ = build_manager()
    key = SessionKey.private(1)

    await manager.record(key, user("uno"), assistant("r1"))
    await manager.record(key, user("dos"), assistant("r2"))

    window = await manager.load_window(key)
    assert [m.content for m in window] == ["uno", "r1", "dos", "r2"]


async def test_ventana_limita_a_los_ultimos_turnos():
    manager, _, _ = build_manager(max_turns=1)
    key = SessionKey.private(1)

    await manager.record(key, user("uno"), assistant("r1"))
    await manager.record(key, user("dos"), assistant("r2"))

    window = await manager.load_window(key)
    assert [m.content for m in window] == ["dos", "r2"]


async def test_ventana_no_empieza_en_un_mensaje_del_asistente():
    manager, repository, _ = build_manager()
    key = SessionKey.private(1)
    repository.sessions[key.value] = SessionState(
        messages=(assistant("suelto"), user("hola"), assistant("qué tal")),
        updated_at=NOW,
    )

    window = await manager.load_window(key)

    assert [m.content for m in window] == ["hola", "qué tal"]


async def test_sesion_expira_tras_la_inactividad():
    manager, repository, clock = build_manager(idle_minutes=240)
    key = SessionKey.private(1)
    await manager.record(key, user("hola"), assistant("qué tal"))

    clock.advance(timedelta(minutes=241))

    assert await manager.load_window(key) == ()
    assert repository.discarded == [key.value]


async def test_sesion_viva_dentro_del_umbral():
    manager, _, clock = build_manager(idle_minutes=240)
    key = SessionKey.private(1)
    await manager.record(key, user("hola"), assistant("qué tal"))

    clock.advance(timedelta(minutes=239))

    assert len(await manager.load_window(key)) == 2


async def test_el_borde_exacto_del_umbral_no_expira():
    manager, _, clock = build_manager(idle_minutes=240)
    key = SessionKey.private(1)
    await manager.record(key, user("hola"), assistant("qué tal"))

    clock.advance(timedelta(minutes=240))

    assert len(await manager.load_window(key)) == 2


async def test_idle_cero_nunca_expira():
    manager, _, clock = build_manager(idle_minutes=0)
    key = SessionKey.private(1)
    await manager.record(key, user("hola"), assistant("qué tal"))

    clock.advance(timedelta(days=30))

    assert len(await manager.load_window(key)) == 2


async def test_n_cero_desactiva_la_memoria():
    manager, repository, _ = build_manager(max_turns=0)
    key = SessionKey.private(1)

    await manager.record(key, user("hola"), assistant("qué tal"))

    assert manager.enabled is False
    assert repository.appended == []
    assert await manager.load_window(key) == ()


async def test_atribucion_con_nombre_en_grupos():
    manager, _, _ = build_manager()
    key = SessionKey.group(-100)

    await manager.record(key, user("¿qué es Kafka?", author="Ana"), assistant("Es..."))

    window = await manager.load_window(key)
    assert [m.content for m in window] == ["Ana: ¿qué es Kafka?", "Es..."]


async def test_sin_atribucion_en_privado():
    manager, _, _ = build_manager()
    key = SessionKey.private(1)

    await manager.record(key, user("hola", author="Ana"), assistant("qué tal"))

    window = await manager.load_window(key)
    assert [m.content for m in window] == ["hola", "qué tal"]


async def test_el_lock_por_clave_es_estable():
    manager, _, _ = build_manager()
    key = SessionKey.private(1)

    assert manager.lock_for(key) is manager.lock_for(key)
    assert manager.lock_for(key) is not manager.lock_for(SessionKey.private(2))


async def test_reset_descarta_la_sesion():
    manager, repository, _ = build_manager()
    key = SessionKey.private(1)
    await manager.record(key, user("hola"), assistant("qué tal"))

    await manager.reset(key)

    assert repository.discarded == [key.value]
    assert await manager.load_window(key) == ()


async def test_la_expiracion_se_registra_sin_contenido(caplog):
    manager, _, clock = build_manager(idle_minutes=240)
    key = SessionKey.private(1)
    await manager.record(key, user("secreto"), assistant("respuesta"))

    clock.advance(timedelta(minutes=241))
    with caplog.at_level(logging.INFO):
        await manager.load_window(key)

    assert "Sesión expirada" in caplog.text
    assert "private:1" in caplog.text
    assert "secreto" not in caplog.text


async def test_el_reset_se_registra_sin_contenido(caplog):
    manager, _, _ = build_manager()
    key = SessionKey.group(-100)

    with caplog.at_level(logging.INFO):
        await manager.reset(key)

    assert "Sesión reiniciada" in caplog.text
    assert "group:-100" in caplog.text


async def test_resumen_se_antepone_al_historial_atribuido_del_grupo():
    manager, repo, _ = build_manager(max_turns=2)
    key = SessionKey.group(-100)
    await manager.record(key, user("uno", "Ana"), assistant("r1"))
    await repo.compact(key, "Ana eligió Kafka")
    await manager.record(key, user("dos", "Luis"), assistant("r2"))
    window = await manager.load_window(key)
    assert window[0].role is MessageRole.ASSISTANT
    assert "Ana eligió Kafka" in window[0].content
    assert [m.content for m in window[1:]] == ["Luis: dos", "r2"]


async def test_expiracion_elimina_tambien_resumen():
    manager, repo, clock = build_manager(max_turns=1)
    key = SessionKey.private(1)
    await manager.record(key, user("uno"), assistant("r1"))
    await repo.compact(key, "resumen")
    clock.advance(timedelta(minutes=241))
    assert await manager.load_window(key) == ()
    assert await repo.load(key) is None
