"""Pruebas del pool de exclusión por sesión en la entrada de Telegram."""

from __future__ import annotations

import asyncio

import pytest

from oraculo.domain.models import SessionKey
from oraculo.telegram.session_locks import SessionUpdateLockPool


async def test_claves_distintas_pueden_ejecutarse_a_la_vez() -> None:
    pool = SessionUpdateLockPool()
    release = asyncio.Event()
    entered_private = asyncio.Event()
    entered_group = asyncio.Event()

    async def hold(key: SessionKey, entered: asyncio.Event) -> None:
        async with pool.hold(key):
            entered.set()
            await release.wait()

    private = asyncio.create_task(hold(SessionKey.private(1), entered_private))
    await asyncio.wait_for(entered_private.wait(), timeout=1)
    group = asyncio.create_task(hold(SessionKey.group(-100), entered_group))

    try:
        await asyncio.wait_for(entered_group.wait(), timeout=1)
    finally:
        release.set()
        await asyncio.gather(private, group)


async def test_misma_clave_espera_y_conserva_el_orden_de_adquisicion() -> None:
    pool = SessionUpdateLockPool()
    key = SessionKey.private(1)
    first_entered = asyncio.Event()
    second_attempted = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()
    order: list[str] = []

    async def first() -> None:
        async with pool.hold(key):
            order.append("primero")
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        second_attempted.set()
        async with pool.hold(key):
            order.append("segundo")
            second_entered.set()

    first_task = asyncio.create_task(first())
    await asyncio.wait_for(first_entered.wait(), timeout=1)
    second_task = asyncio.create_task(second())
    await asyncio.wait_for(second_attempted.wait(), timeout=1)

    assert order == ["primero"]
    assert not second_entered.is_set()

    release_first.set()
    await asyncio.wait_for(second_entered.wait(), timeout=1)
    await asyncio.gather(first_task, second_task)

    assert order == ["primero", "segundo"]


async def test_error_y_cancelacion_no_dejan_lock_bloqueado() -> None:
    pool = SessionUpdateLockPool()
    key = SessionKey.group(-100)
    owner_entered = asyncio.Event()
    release_owner = asyncio.Event()
    waiter_attempted = asyncio.Event()

    async def owner() -> None:
        async with pool.hold(key):
            owner_entered.set()
            await release_owner.wait()
            raise RuntimeError("fallo esperado")

    async def waiter() -> None:
        waiter_attempted.set()
        async with pool.hold(key):
            raise AssertionError("el waiter cancelado no debe entrar")

    owner_task = asyncio.create_task(owner())
    await asyncio.wait_for(owner_entered.wait(), timeout=1)
    waiter_task = asyncio.create_task(waiter())
    await asyncio.wait_for(waiter_attempted.wait(), timeout=1)
    waiter_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiter_task

    release_owner.set()
    with pytest.raises(RuntimeError, match="fallo esperado"):
        await owner_task

    async with pool.hold(key):
        pass
