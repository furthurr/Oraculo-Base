"""Tests del intervalo de edición del streaming de Pi (spec modo-pi/streaming-intervalo)."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from datetime import timedelta
from types import SimpleNamespace

import pytest
from telegram.error import RetryAfter, TelegramError

from oraculo.application.pi_mode import PiDisplayEvent
from oraculo.domain.errors import ConfigError
from oraculo.domain.pi import PiAccess, PiSessionState
from oraculo.infrastructure.config import load_config
import oraculo.telegram.handlers as handlers
from oraculo.telegram.handlers import OraculoMessageHandler

BASE_ENV = {"TELEGRAM_BOT_TOKEN": "123:abc", "AI_PROVIDER": "deepseek", "DEEPSEEK_API_KEY": "sk-test"}


# --- Config (Req 2.x) -------------------------------------------------------


def test_intervalo_por_defecto_es_un_segundo() -> None:
    assert load_config(BASE_ENV).pi_stream_edit_interval_s == 1.0


@pytest.mark.parametrize("valor", ["", "   "])
def test_intervalo_vacio_usa_default(valor: str) -> None:
    env = {**BASE_ENV, "PI_STREAM_EDIT_INTERVAL_S": valor}
    assert load_config(env).pi_stream_edit_interval_s == 1.0


def test_intervalo_personalizado() -> None:
    env = {**BASE_ENV, "PI_STREAM_EDIT_INTERVAL_S": "0.4"}
    assert load_config(env).pi_stream_edit_interval_s == 0.4


@pytest.mark.parametrize("valor", ["rapido", "0", "-1"])
def test_intervalo_invalido_falla_sin_revelar_valor(valor: str) -> None:
    env = {**BASE_ENV, "PI_STREAM_EDIT_INTERVAL_S": valor}
    with pytest.raises(ConfigError) as info:
        load_config(env)
    assert "PI_STREAM_EDIT_INTERVAL_S" in str(info.value)
    assert valor not in str(info.value)


# --- Handler (Req 1.x) ------------------------------------------------------


class FakeSentMessage:
    """Mensaje de Telegram ya enviado: registra ediciones y puede fallar."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.edits: list[str] = []
        self.edit_attempts: list[float] = []
        self.failures: list[TelegramError] = []

    async def edit_text(self, text: str) -> None:
        self.edit_attempts.append(time.monotonic())
        if self.failures:
            raise self.failures.pop(0)
        self.edits.append(text)
        self.text = text


class FakeIncomingMessage:
    def __init__(self, text: str) -> None:
        self.text = text
        self.entities: tuple[object, ...] = ()
        self.caption: str | None = None
        self.caption_entities: tuple[object, ...] = ()
        self.reply_to_message: object = None
        self.message_thread_id: int | None = None
        self.sent: list[FakeSentMessage] = []
        self.sent_at: list[float] = []

    def parse_entity(self, entity: object) -> str:
        return ""

    def parse_caption_entity(self, entity: object) -> str:
        return ""

    async def reply_text(self, text: str, **_: object) -> FakeSentMessage:
        sent = FakeSentMessage(text)
        self.sent.append(sent)
        self.sent_at.append(time.monotonic())
        return sent


class ScriptedPiService:
    """Servicio Pi falso: emite eventos con pausas `(espera_s, evento)`."""

    def __init__(self, script: list[tuple[float, PiDisplayEvent]]) -> None:
        self._script = script

    def is_in_pi_mode(self, user_id: int) -> bool:
        return True

    def get_state(self, user_id: int) -> PiSessionState:
        return PiSessionState.PI_ACTIVO

    async def handle(
        self, user_id: int, text: str, *, images: list[bytes] | None = None
    ) -> AsyncIterator[PiDisplayEvent]:
        for wait_s, event in self._script:
            if wait_s:
                await asyncio.sleep(wait_s)
            yield event


class AllowAll:
    def pi_access(self, message: object) -> PiAccess:
        return PiAccess(allowed=True, reply=None)


def delta(text: str) -> PiDisplayEvent:
    return PiDisplayEvent(type="text_delta", content=text)


def final(text: str) -> PiDisplayEvent:
    return PiDisplayEvent(type="text_final", content=text)


def notice(text: str) -> PiDisplayEvent:
    return PiDisplayEvent(type="tool_notice", content=text)


async def run(script: list[tuple[float, PiDisplayEvent]], interval_s: float) -> FakeIncomingMessage:
    handler = OraculoMessageHandler(
        service=AllowAll(),  # type: ignore[arg-type]
        pi_mode_service=ScriptedPiService(script),  # type: ignore[arg-type]
        typing_start_delay_s=10.0,
        pi_stream_edit_interval_s=interval_s,
    )
    message = FakeIncomingMessage("hola")
    update = SimpleNamespace(
        effective_message=message,
        effective_chat=SimpleNamespace(id=100, type="private"),
        effective_user=SimpleNamespace(id=1, is_bot=False, first_name="Ana", username=None),
    )
    bot = SimpleNamespace(username="oraculo_bot", id=424242)
    await handler(update, SimpleNamespace(bot=bot))  # type: ignore[arg-type]
    return message


async def test_fragmentos_dentro_del_intervalo_no_editan() -> None:
    message = await run(
        [(0, delta("Ho")), (0, delta("la")), (0, delta(" mun")), (0, delta("do"))],
        interval_s=10.0,
    )
    assert len(message.sent) == 1
    assert message.sent[0].text == "🤖💻 Ho"
    assert message.sent[0].edits == []


async def test_fragmento_tras_el_intervalo_edita_con_todo_lo_acumulado() -> None:
    message = await run(
        [(0, delta("Ho")), (0, delta("la")), (0.08, delta(" mundo"))],
        interval_s=0.05,
    )
    assert message.sent[0].edits == ["🤖💻 Hola mundo"]


async def test_final_se_muestra_de_inmediato() -> None:
    message = await run(
        [(0, delta("Ho")), (0, delta("la")), (0, final("Hola mundo"))],
        interval_s=10.0,
    )
    assert message.sent[0].edits == ["🤖💻 Hola mundo"]


async def test_final_igual_al_visible_no_repite_edicion() -> None:
    message = await run([(0, delta("Hola")), (0, final("Hola"))], interval_s=10.0)
    assert message.sent[0].edits == []


def test_intervalo_edicion_aviso_herramienta_es_dos_segundos() -> None:
    assert handlers.PI_TOOL_NOTICE_EDIT_INTERVAL_S == 2.0


def test_aviso_repetido_respeta_limite_de_caracteres_de_telegram() -> None:
    content = "x" * (handlers.TELEGRAM_MAX_CHARS - 1)
    rendered = handlers._render_tool_notice(content, 3)
    assert len(rendered) == handlers.TELEGRAM_MAX_CHARS
    assert "3 repeticiones" in rendered


async def test_repeticiones_se_acumulan_y_se_vacian_antes_del_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handlers, "PI_TOOL_NOTICE_EDIT_INTERVAL_S", 0.04)
    status = "🤖💻 Ejecutando comandos…"
    message = await run(
        [(0, notice(status)), (0, notice(status)), (0, notice(status)), (0, notice(status)),
         (0, final("listo"))],
        interval_s=10.0,
    )
    assert [sent.text for sent in message.sent] == [status + " …" * 3, "🤖💻 listo"]
    assert message.sent[0].edits == [status + " …" * 3]


async def test_ediciones_del_mismo_aviso_separadas_por_el_intervalo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interval = 0.04
    monkeypatch.setattr(handlers, "PI_TOOL_NOTICE_EDIT_INTERVAL_S", interval)
    status = "🤖💻 Leyendo archivos…"
    message = await run(
        [(0, notice(status)), (0, notice(status)), (0, notice(status)),
         (interval + 0.01, notice(status)), (0, final("listo"))],
        interval_s=10.0,
    )
    edits = message.sent[0].edit_attempts
    assert len(message.sent[0].edits) == 2
    assert edits[1] - edits[0] >= interval
    assert message.sent[0].edits == [status + " …" * 2, status + " …" * 3]


async def test_aviso_distinto_se_envia_inmediatamente_y_preserva_pendientes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interval = 0.04
    monkeypatch.setattr(handlers, "PI_TOOL_NOTICE_EDIT_INTERVAL_S", interval)
    commands = "🤖💻 Ejecutando comandos…"
    reading = "🤖💻 Leyendo archivos…"
    message = await run(
        [(0, notice(commands)), (0, notice(commands)), (0, notice(reading)),
         (0, final("listo"))],
        interval_s=10.0,
    )
    assert [sent.text for sent in message.sent] == [commands + " …", reading, "🤖💻 listo"]
    assert message.sent_at[1] - message.sent_at[0] < interval
    assert message.sent[0].edits == [commands + " …"]


async def test_repeticiones_pendientes_se_vacian_antes_del_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handlers, "PI_TOOL_NOTICE_EDIT_INTERVAL_S", 0.01)
    status = "🤖💻 Ejecutando comandos…"
    message = await run(
        [(0, notice(status)), (0, notice(status)),
         (0, PiDisplayEvent(type="error", content="🤖💻 fallo"))],
        interval_s=10.0,
    )
    assert [sent.text for sent in message.sent] == [status + " …", "🤖💻 fallo"]


async def test_retry_after_conserva_repeticiones_pendientes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(handlers, "PI_TOOL_NOTICE_EDIT_INTERVAL_S", 0.01)
    original = FakeSentMessage.__init__

    def retry_once(self: FakeSentMessage, text: str) -> None:
        original(self, text)
        self.failures = [RetryAfter(timedelta(seconds=0.01))]

    monkeypatch.setattr(FakeSentMessage, "__init__", retry_once)
    status = "🤖💻 Ejecutando comandos…"
    message = await run(
        [(0, notice(status)), (0, notice(status)), (0, final("listo"))],
        interval_s=10.0,
    )
    assert message.sent[0].edits == [status + " …"]
    assert len(message.sent[0].edit_attempts) == 2
    assert message.sent[0].edit_attempts[1] - message.sent[0].edit_attempts[0] >= 0.01


async def test_retry_after_intermedio_se_registra_y_pospone(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = [
        (0, delta("a")),
        (0.03, delta("b")),   # edición falla con RetryAfter(1 s)
        (0.03, delta("c")),   # dentro del plazo de Telegram: no se edita
        (0, final("abc")),
    ]
    original = FakeSentMessage.__init__

    def failing_init(self: FakeSentMessage, text: str) -> None:
        original(self, text)
        self.failures = [RetryAfter(timedelta(seconds=1))]

    monkeypatch.setattr(FakeSentMessage, "__init__", failing_init)
    with caplog.at_level(logging.WARNING, logger="oraculo.telegram.handlers"):
        message = await run(script, interval_s=0.01)
    # Solo la edición final llega; la intermedia se descartó y no se reintentó antes de tiempo.
    assert message.sent[0].edits == ["🤖💻 abc"]
    assert any("flood" in r.getMessage().lower() for r in caplog.records)


async def test_retry_after_en_final_espera_y_reintenta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        await real_sleep(0)

    original = FakeSentMessage.__init__

    def failing_init(self: FakeSentMessage, text: str) -> None:
        original(self, text)
        self.failures = [RetryAfter(3)]

    monkeypatch.setattr(FakeSentMessage, "__init__", failing_init)
    monkeypatch.setattr("oraculo.telegram.handlers.asyncio.sleep", fake_sleep)
    message = await run([(0, delta("Ho")), (0, final("Hola"))], interval_s=10.0)
    assert message.sent[0].edits == ["🤖💻 Hola"]
    assert 3.0 in sleeps


async def test_otro_error_de_telegram_se_registra_sin_cortar_el_turno(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = FakeSentMessage.__init__
    created: list[FakeSentMessage] = []

    def failing_init(self: FakeSentMessage, text: str) -> None:
        original(self, text)
        if not created:  # solo falla el primer mensaje del turno
            self.failures = [TelegramError("boom")]
        created.append(self)

    monkeypatch.setattr(FakeSentMessage, "__init__", failing_init)
    with caplog.at_level(logging.WARNING, logger="oraculo.telegram.handlers"):
        message = await run(
            [(0, delta("a")), (0, final("ab")), (0, delta("c")), (0, final("cd"))],
            interval_s=10.0,
        )
    assert message.sent[1].edits == ["🤖💻 cd"]
    assert any("chat_id=100" in r.getMessage() for r in caplog.records)


async def test_build_application_propaga_el_intervalo() -> None:
    from oraculo.telegram.bot import build_application

    async def noop(_: object) -> None:
        return None

    app = build_application(
        "123:abc",
        AllowAll(),  # type: ignore[arg-type]
        pi_stream_edit_interval_s=0.7,
        post_shutdown=noop,  # type: ignore[arg-type]
    )
    callbacks = [h.callback for group in app.handlers.values() for h in group]
    handler = next(c for c in callbacks if isinstance(c, OraculoMessageHandler))
    assert handler._pi_stream_edit_interval_s == 0.7
