"""Tests del almacenamiento temporal de adjuntos (Wave 2, tarea 2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from oraculo.domain.attachments import AttachmentMode
from oraculo.domain.errors import AttachmentError
from oraculo.domain.models import AttachmentKey, ChatKind
from oraculo.infrastructure.temp_attachments import TempAttachmentStore

KEY = AttachmentKey(kind=ChatKind.PRIVATE, chat_id=7, user_id=7)
OTHER = AttachmentKey(kind=ChatKind.PRIVATE, chat_id=8, user_id=8)


def store_at(tmp_path: Path) -> TempAttachmentStore:
    return TempAttachmentStore(root=tmp_path / "temp" / "attachments")


async def test_store_y_load_conservan_los_bytes(tmp_path: Path):
    store = store_at(tmp_path)

    await store.store(KEY, "a.txt", "text/plain", AttachmentMode.TEXT, b"hola")

    assert await store.load(KEY, "a.txt") == b"hola"
    assert await store.list(KEY) == ("a.txt",)


async def test_nombres_con_traversal_se_sanean(tmp_path: Path):
    store = store_at(tmp_path)

    await store.store(KEY, "../../etc/passwd", "text/plain", AttachmentMode.TEXT, b"x")

    names = await store.list(KEY)
    assert names == ("passwd",)
    # Nada fuera de la raíz del store.
    assert not (tmp_path / "etc").exists()
    assert await store.load(KEY, "passwd") == b"x"


async def test_nombres_con_separadores_se_sanean(tmp_path: Path):
    store = store_at(tmp_path)

    await store.store(KEY, "sub/dir\\a.txt", "text/plain", AttachmentMode.TEXT, b"x")

    assert await store.list(KEY) == ("a.txt",)


async def test_load_de_nombre_inexistente_lanza_attachment_error(tmp_path: Path):
    store = store_at(tmp_path)

    with pytest.raises(AttachmentError) as exc:
        await store.load(KEY, "no-existe.txt")
    assert exc.value.reason == "not_found"


async def test_load_de_nombre_malicioso_lanza_attachment_error(tmp_path: Path):
    store = store_at(tmp_path)

    with pytest.raises(AttachmentError) as exc:
        await store.load(KEY, "../secreto.txt")
    assert exc.value.reason == "invalid_name"


async def test_delete_es_idempotente(tmp_path: Path):
    store = store_at(tmp_path)
    await store.store(KEY, "a.txt", "text/plain", AttachmentMode.TEXT, b"x")

    await store.delete(KEY, "a.txt")
    await store.delete(KEY, "a.txt")

    assert await store.list(KEY) == ()


async def test_clear_solo_afecta_a_una_sesion(tmp_path: Path):
    store = store_at(tmp_path)
    await store.store(KEY, "a.txt", "text/plain", AttachmentMode.TEXT, b"x")
    await store.store(OTHER, "b.txt", "text/plain", AttachmentMode.TEXT, b"y")

    await store.clear(KEY)

    assert await store.list(KEY) == ()
    assert await store.list(OTHER) == ("b.txt",)


async def test_clear_all_vacia_todas_las_sesiones(tmp_path: Path):
    store = store_at(tmp_path)
    await store.store(KEY, "a.txt", "text/plain", AttachmentMode.TEXT, b"x")
    await store.store(OTHER, "b.txt", "text/plain", AttachmentMode.TEXT, b"y")

    await store.clear_all()

    assert await store.list(KEY) == ()
    assert await store.list(OTHER) == ()


async def test_sesiones_de_grupo_y_privado_no_se_mezclan(tmp_path: Path):
    store = store_at(tmp_path)
    private = AttachmentKey(kind=ChatKind.PRIVATE, chat_id=1, user_id=1)
    group = AttachmentKey(kind=ChatKind.GROUP, chat_id=-100, user_id=1)

    await store.store(private, "a.txt", "text/plain", AttachmentMode.TEXT, b"x")

    assert await store.list(private) == ("a.txt",)
    assert await store.list(group) == ()


async def test_miembros_del_mismo_grupo_guardan_adjunto_en_almacenes_distintos(
    tmp_path: Path,
):
    store = store_at(tmp_path)
    ana = AttachmentKey(kind=ChatKind.GROUP, chat_id=-100, user_id=1)
    luis = AttachmentKey(kind=ChatKind.GROUP, chat_id=-100, user_id=2)

    await store.store(ana, "nota.txt", "text/plain", AttachmentMode.TEXT, b"Ana")
    await store.store(luis, "nota.txt", "text/plain", AttachmentMode.TEXT, b"Luis")

    assert await store.load(ana, "nota.txt") == b"Ana"
    assert await store.load(luis, "nota.txt") == b"Luis"
