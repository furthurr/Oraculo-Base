"""Configuración de logging (sin exponer secretos)."""

from __future__ import annotations

import logging

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

# `httpx` y `httpcore` registran la URL completa de cada petición a nivel INFO.
# En la API de Telegram esa URL incluye el token del bot, así que estos loggers
# nunca deben emitir por debajo de WARNING (Req 4.4).
_QUIET_LOGGERS = ("httpx", "httpcore")


def configure_logging(level: int = logging.INFO) -> None:
    """Configura el logging raíz una sola vez, sin filtrar credenciales."""
    logging.basicConfig(level=level, format=_LOG_FORMAT)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
