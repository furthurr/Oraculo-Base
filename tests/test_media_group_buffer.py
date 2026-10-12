"""Tests del buffer de álbumes de Telegram (Wave 4, tarea 4.1)."""

from __future__ import annotations

import asyncio

import pytest

from oraculo.domain.attachments import IncomingAttachment
from oraculo.telegram.media_groups import MediaGroupBuffer


def incoming(name: str) -> IncomingAttachment:
    return IncomingAttachment(
        file_id=f"file-{name}", file_name=name,
        mime_type="application/octet-stream", file_size=10, kind="document",
    )


async def test_mensajes_del_mismo_album_se_agrupan():
    buffer = MediaGroupBuffer(window_s=0.05)

    first = buffer.add(chat_id=1, media_group_id="g1", attachment=incoming("a.txt"))
    second = buffer.add(chat_id=1, media_group_id="g1", attachment=incoming("b.txt"))

    assert first is None
    assert second is None

    await asyncio.sleep(0.08)
    groups = buffer.flush_ready()

    assert len(groups) == 1
    chat_id, _, attachments = groups[0]
    assert chat_id == 1
    assert [a.file_name for a in attachments] == ["a.txt", "b.txt"]


async def test_mensaje_sin_album_sale_inmediato():
    buffer = MediaGroupBuffer(window_s=0.05)

    ready = buffer.add(chat_id=1, media_group_id=None, attachment=incoming("solo.txt"))

    assert ready is not None
    assert [a.file_name for a in ready] == ["solo.txt"]


async def test_albumes_de_chats_distintos_no_se_mezclan():
    buffer = MediaGroupBuffer(window_s=0.05)

    buffer.add(chat_id=1, media_group_id="g1", attachment=incoming("a.txt"))
    buffer.add(chat_id=2, media_group_id="g1", attachment=incoming("b.txt"))

    await asyncio.sleep(0.08)
    groups = buffer.flush_ready()

    assert sorted(chat_id for chat_id, _, _ in groups) == [1, 2]


async def test_el_orden_de_llegada_se_conserva():
    buffer = MediaGroupBuffer(window_s=0.05)

    for name in ("z.txt", "a.txt", "m.txt"):
        buffer.add(chat_id=1, media_group_id="g1", attachment=incoming(name))

    await asyncio.sleep(0.08)
    _, _, attachments = buffer.flush_ready()[0]

    assert [a.file_name for a in attachments] == ["z.txt", "a.txt", "m.txt"]


async def test_flush_ready_no_vuelve_a_entregar_lo_mismo():
    buffer = MediaGroupBuffer(window_s=0.05)
    buffer.add(chat_id=1, media_group_id="g1", attachment=incoming("a.txt"))

    await asyncio.sleep(0.08)
    assert len(buffer.flush_ready()) == 1
    assert buffer.flush_ready() == []


async def test_ventana_corta_no_espera_mas_de_la_cuenta():
    buffer = MediaGroupBuffer(window_s=0.05)
    buffer.add(chat_id=1, media_group_id="g1", attachment=incoming("a.txt"))

    # Antes de la ventana no sale; después sí.
    assert buffer.flush_ready() == []
    await asyncio.sleep(0.08)
    assert len(buffer.flush_ready()) == 1


def test_vaciado_individual_no_drena_otros_albumes_vencidos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [0.0]
    monkeypatch.setattr(
        "oraculo.telegram.media_groups.time.monotonic", lambda: clock[0]
    )
    buffer = MediaGroupBuffer(window_s=5)
    buffer.add(chat_id=-100, media_group_id="album-a", attachment=incoming("a1"))
    buffer.add(chat_id=-100, media_group_id="album-b", attachment=incoming("b1"))
    clock[0] = 6.0

    first = buffer.flush_group(chat_id=-100, media_group_id="album-a")
    buffer.add(chat_id=-100, media_group_id="album-b", attachment=incoming("b2"))
    second = buffer.flush_group(chat_id=-100, media_group_id="album-b")

    assert first == (-100, "album-a", (incoming("a1"),))
    assert second == (-100, "album-b", (incoming("b1"), incoming("b2")))
