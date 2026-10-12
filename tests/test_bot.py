"""Tests del registro de handlers de Telegram (Req 4.4, 4.7)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import pytest
from telegram import BotCommand
from telegram.ext import Application
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler as PTBMessageHandler

from oraculo.application.conversation import ConversationService
from oraculo.domain.models import IncomingMessage
from oraculo.telegram.bot import build_application
from oraculo.telegram.session_locks import SessionUpdateLockPool
from oraculo.telegram.handlers import OraculoCommandHandler, OraculoMessageHandler


class DummyService(ConversationService):
    def __init__(self) -> None:
        pass

    async def handle(self, message: IncomingMessage) -> str | None:
        del message
        return None

    async def handle_reset(self, message: IncomingMessage) -> str | None:
        del message
        return None


def test_los_comandos_se_registran_antes_que_los_mensajes():
    async def close(application: Application) -> None:
        del application

    application = build_application(
        "123:abc", DummyService(), post_shutdown=close
    )

    handlers = application.handlers[0]
    assert application.concurrent_updates == 1
    assert isinstance(handlers[0], CommandHandler)
    assert set(handlers[0].commands) == {"new", "reset", "help"}
    assert isinstance(handlers[1], PTBMessageHandler)


def test_limite_global_y_pool_se_comparten_entre_handlers() -> None:
    async def close(application: Application) -> None:
        del application

    locks = SessionUpdateLockPool()
    application = build_application(
        "123:abc",
        DummyService(),
        max_concurrent_updates=3,
        session_update_locks=locks,
        post_shutdown=close,
    )

    callbacks = [handler.callback for group in application.handlers.values() for handler in group]
    command_handler = next(
        callback for callback in callbacks if isinstance(callback, OraculoCommandHandler)
    )
    message_handler = next(
        callback for callback in callbacks if isinstance(callback, OraculoMessageHandler)
    )

    assert application.concurrent_updates == 3
    assert command_handler._session_update_locks is locks
    assert message_handler._session_update_locks is locks


async def test_registra_menu_de_comandos_en_post_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def close(application: Application) -> None:
        del application

    application = build_application(
        "123:abc", DummyService(), post_shutdown=close
    )
    registered: list[BotCommand] = []

    async def set_my_commands(bot: object, commands: list[BotCommand]) -> None:
        del bot
        registered.extend(commands)

    monkeypatch.setattr(type(application.bot), "set_my_commands", set_my_commands)
    assert application.post_init is not None
    await application.post_init(application)

    assert [command.command for command in registered] == [
        "new", "agent", "exitagent", "help"
    ]
    assert registered[1].description == "Elegir proyecto y activar el modo agente"
    assert registered[2].description == "Salir del modo agente"


def test_registra_callback_de_cierre_en_post_shutdown():
    async def close(application: Application) -> None:
        del application

    callback: Callable[[Application], Awaitable[None]] = close
    application = build_application(
        "123:abc", DummyService(), post_shutdown=callback
    )

    assert application.post_shutdown is callback
