"""Construcción de la aplicación de python-telegram-bot."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from telegram import BotCommand
from telegram.error import TelegramError
from telegram.ext import Application, ApplicationBuilder, CommandHandler, filters
from telegram.ext import MessageHandler as PTBMessageHandler

from oraculo.application.conversation import ConversationService
from oraculo.application.pi_mode import PiModeService
from oraculo.telegram.handlers import (
    PI_STREAM_EDIT_INTERVAL_S,
    OraculoCommandHandler,
    OraculoMessageHandler,
)
from oraculo.telegram.session_locks import SessionUpdateLockPool

logger = logging.getLogger(__name__)

TELEGRAM_COMMANDS = (
    BotCommand("new", "Iniciar una conversación nueva"),
    BotCommand("agent", "Elegir proyecto y activar el modo agente"),
    BotCommand("exitagent", "Salir del modo agente"),
    BotCommand("help", "Mostrar ayuda"),
)


def build_application(
    token: str,
    service: ConversationService,
    *,
    pi_mode_service: PiModeService | None = None,
    pi_stream_edit_interval_s: float = PI_STREAM_EDIT_INTERVAL_S,
    max_concurrent_updates: int = 1,
    session_update_locks: SessionUpdateLockPool | None = None,
    post_shutdown: Callable[[Application], Awaitable[None]],
    post_init: Callable[[Application], Awaitable[None]] | None = None,
) -> Application:
    """Crea la aplicación con los comandos de sesión y el handler de mensajes.

    El `CommandHandler` va primero: los comandos se interceptan antes del
    `MessageHandler`, de modo que nunca llegan a la IA ni se guardan.
    """
    update_locks = session_update_locks or SessionUpdateLockPool()
    builder = (
        ApplicationBuilder()
        .token(token)
        .concurrent_updates(max_concurrent_updates)
        .post_shutdown(post_shutdown)
    )

    async def initialize(application: Application) -> None:
        try:
            await application.bot.set_my_commands(list(TELEGRAM_COMMANDS))
        except TelegramError:
            logger.warning("No se pudo registrar el menú de comandos de Telegram")
        if post_init is not None:
            await post_init(application)

    application = builder.post_init(initialize).build()
    application.add_handler(
        CommandHandler(
            ["new", "reset", "help"],
            OraculoCommandHandler(service, session_update_locks=update_locks),
        )
    )
    application.add_handler(
        PTBMessageHandler(
            filters.ALL,
            OraculoMessageHandler(
                service,
                pi_mode_service=pi_mode_service,
                pi_stream_edit_interval_s=pi_stream_edit_interval_s,
                session_update_locks=update_locks,
            ),
        )
    )
    return application
