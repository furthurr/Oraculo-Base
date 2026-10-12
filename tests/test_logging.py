"""Tests de configuración de logging (Req 4.4 y bugfix de fuga del token).

Regresión: `httpx` no debe emitir la URL de las peticiones a nivel INFO porque la
URL de la API de Telegram contiene el token del bot.
"""

from __future__ import annotations

import logging

import pytest

from oraculo.infrastructure.logging import configure_logging

URL_CON_TOKEN = "POST https://api.telegram.org/bot8758280678:AAGsecreto/getMe"


@pytest.mark.parametrize("logger_name", ["httpx", "httpcore"])
def test_httpx_no_emite_en_info(logger_name, caplog):
    configure_logging()

    with caplog.at_level(logging.INFO):
        logging.getLogger(logger_name).info("HTTP Request: %s", URL_CON_TOKEN)

    assert "AAGsecreto" not in caplog.text


@pytest.mark.parametrize("logger_name", ["httpx", "httpcore"])
def test_nivel_de_los_loggers_silenciados(logger_name):
    configure_logging()

    assert logging.getLogger(logger_name).getEffectiveLevel() >= logging.WARNING


def test_la_aplicacion_sigue_emitiendo_en_info(caplog):
    configure_logging()

    with caplog.at_level(logging.INFO):
        logging.getLogger("oraculo.application.conversation").info("consulta atendida")

    assert "consulta atendida" in caplog.text
