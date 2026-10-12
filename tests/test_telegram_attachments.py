"""Tests de normalización de adjuntos desde Telegram (Wave 4, tarea 4.2)."""

from __future__ import annotations

from types import SimpleNamespace

from oraculo.domain.attachments import IncomingAttachment
from oraculo.telegram.attachments import normalize_attachments


def document(file_id="doc-1", name="a.txt", mime="text/plain", size=10):
    return SimpleNamespace(
        file_id=file_id, file_name=name, mime_type=mime, file_size=size,
    )


def photo(file_id="ph-1", size=10):
    return SimpleNamespace(file_id=file_id, file_size=size)


def media(**kwargs):
    return SimpleNamespace(**kwargs)


def test_documento_se_normaliza_a_incoming_attachment():
    message = media(document=document(), photo=None, audio=None,
                    video=None, voice=None, animation=None)

    attachments = normalize_attachments(message)

    assert len(attachments) == 1
    attachment = attachments[0]
    assert isinstance(attachment, IncomingAttachment)
    assert attachment.file_id == "doc-1"
    assert attachment.file_name == "a.txt"
    assert attachment.mime_type == "text/plain"
    assert attachment.file_size == 10
    assert attachment.kind == "document"


def test_photo_se_normaliza_con_el_tamano_mayor():
    small = photo(file_id="ph-small", size=10)
    large = photo(file_id="ph-large", size=99)
    message = media(document=None, photo=[small, large], audio=None,
                    video=None, voice=None, animation=None)

    attachments = normalize_attachments(message)

    assert len(attachments) == 1
    assert attachments[0].file_id == "ph-large"
    assert attachments[0].kind == "photo"


def test_audio_video_voice_y_animation_se_normalizan():
    message = media(
        document=None, photo=None,
        audio=media(file_id="au-1", file_name="a.mp3", mime_type="audio/mpeg", file_size=1),
        video=media(file_id="vi-1", file_name="v.mp4", mime_type="video/mp4", file_size=2),
        voice=media(file_id="vo-1", file_name=None, mime_type="audio/ogg", file_size=3),
        animation=media(file_id="an-1", file_name="g.gif", mime_type="image/gif", file_size=4),
    )

    attachments = normalize_attachments(message)

    assert [(a.kind, a.file_id) for a in attachments] == [
        ("audio", "au-1"), ("video", "vi-1"), ("voice", "vo-1"), ("animation", "an-1"),
    ]


def test_mensaje_sin_archivos_devuelve_vacio():
    message = media(document=None, photo=None, audio=None,
                    video=None, voice=None, animation=None)

    assert normalize_attachments(message) == ()


def test_varios_tipos_en_un_mensaje_se_conservan_en_orden():
    message = media(
        document=document(file_id="doc-1", name="a.txt"),
        photo=[photo(file_id="ph-1", size=5)],
        audio=None, video=None, voice=None, animation=None,
    )

    attachments = normalize_attachments(message)

    assert [a.file_id for a in attachments] == ["doc-1", "ph-1"]


# --- Descarga con validación previa de peso --------------------------------

class FakeFile:
    def __init__(self, data: bytes):
        self._data = data

    async def download_as_bytearray(self):
        return bytearray(self._data)


class FakeBot:
    def __init__(self, data: bytes = b"contenido", fail: bool = False):
        self._data = data
        self._fail = fail
        self.requested: list[str] = []

    async def get_file(self, file_id):
        self.requested.append(file_id)
        if self._fail:
            raise RuntimeError("getFile no disponible")
        return FakeFile(self._data)


async def test_descarga_devuelve_los_bytes():
    from oraculo.telegram.attachments import download_attachment

    bot = FakeBot(b"hola")
    attachment = IncomingAttachment(
        file_id="doc-1", file_name="a.txt", mime_type="text/plain",
        file_size=4, kind="document",
    )

    data = await download_attachment(bot, attachment)

    assert data == b"hola"
    assert bot.requested == ["doc-1"]


async def test_sobre_peso_declarado_no_se_descarga():
    from oraculo.telegram.attachments import download_attachment

    bot = FakeBot()
    attachment = IncomingAttachment(
        file_id="doc-1", file_name="grande.bin",
        mime_type="application/octet-stream",
        file_size=20 * 1024 * 1024 + 1, kind="document",
    )

    data = await download_attachment(bot, attachment)

    assert data is None
    assert bot.requested == []


async def test_descarga_real_sobre_el_limite_llega_a_la_aplicacion_para_rechazo():
    from oraculo.telegram.attachments import download_attachment

    bot = FakeBot(b"demasiado")
    attachment = IncomingAttachment(
        file_id="audio-1", file_name=None, mime_type=None,
        file_size=None, kind="voice",
    )

    data = await download_attachment(bot, attachment, max_bytes=3)

    assert data == b"demasiado"


async def test_error_de_descarga_devuelve_none():
    from oraculo.telegram.attachments import download_attachment

    bot = FakeBot(fail=True)
    bot.get_file = failing_get_file
    attachment = IncomingAttachment(
        file_id="doc-1", file_name="a.txt", mime_type="text/plain",
        file_size=4, kind="document",
    )

    assert await download_attachment(bot, attachment) is None


async def failing_get_file(file_id):
    from telegram.error import TelegramError

    raise TelegramError("getFile no disponible")


async def test_error_de_descarga_no_registra_file_id(caplog):
    from oraculo.telegram.attachments import download_attachment

    bot = FakeBot()
    bot.get_file = failing_get_file
    attachment = IncomingAttachment(
        file_id="opaque-telegram-id", file_name=None, mime_type=None,
        file_size=4, kind="voice",
    )

    assert await download_attachment(bot, attachment) is None

    assert "opaque-telegram-id" not in caplog.text
