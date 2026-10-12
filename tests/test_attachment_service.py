"""Tests del servicio de aplicación de adjuntos (Wave 3)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from oraculo.application.attachments import (
    AttachmentService,
    MAX_FILE_BYTES,
    NextAction,
)
from oraculo.domain.attachments import (
    AttachmentMode,
    IncomingAttachment,
    StoredAttachment,
)
from oraculo.domain.errors import AttachmentError
from oraculo.domain.models import AttachmentKey, ChatKind

KEY = AttachmentKey(kind=ChatKind.PRIVATE, chat_id=7, user_id=7)


class FakeAttachmentStore:
    """Almacenamiento en memoria de adjuntos por sesión."""

    def __init__(self) -> None:
        self.files: dict[str, dict[str, bytes]] = {}
        self.cleared: list[str] = []
        self.cleared_all = 0

    async def store(self, key, name, mime_type, mode, data):  # type: ignore[no-untyped-def]
        self.files.setdefault(key.value, {})[name] = data

    async def load(self, key, name):  # type: ignore[no-untyped-def]
        try:
            return self.files[key.value][name]
        except KeyError:
            raise AttachmentError("not_found") from None

    async def list(self, key):  # type: ignore[no-untyped-def]
        return tuple(self.files.get(key.value, {}))

    async def delete(self, key, name):  # type: ignore[no-untyped-def]
        self.files.get(key.value, {}).pop(name, None)

    async def clear(self, key):  # type: ignore[no-untyped-def]
        self.cleared.append(key.value)
        self.files.pop(key.value, None)

    async def clear_all(self):  # type: ignore[no-untyped-def]
        self.cleared_all += 1
        self.files.clear()


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.now

    def advance_minutes(self, minutes: int) -> None:
        from datetime import timedelta

        self.now += timedelta(minutes=minutes)


def incoming(name: str, size: int = 10) -> IncomingAttachment:
    return IncomingAttachment(
        file_id=f"file-{name}", file_name=name,
        mime_type="application/octet-stream", file_size=size,
        kind="document",
    )


def make_service(
    store: FakeAttachmentStore | None = None,
    *,
    binary_ok: set[str] | None = None,
) -> tuple[AttachmentService, FakeAttachmentStore]:
    from oraculo.infrastructure.text_extractor import TextExtractor

    store = store or FakeAttachmentStore()
    allowed = binary_ok or set()

    def binary_supported(name: str, mime_type: str) -> bool:
        return name in allowed

    service = AttachmentService(
        store=store,
        binary_supported=binary_supported,
        clock=FakeClock(),
        extract_text=TextExtractor(max_chars=1000).extract,
    )
    return service, store


# --- Tarea 3.1: límite de 5 por interacción ---------------------------------

async def test_mas_de_cinco_archivos_toma_los_cinco_primeros_y_avisa():
    service, store = make_service()
    items = [(incoming(f"f{i}.txt"), b"contenido") for i in range(7)]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert outcome.action is NextAction.PROCESS
    assert len(outcome.accepted) == 5
    assert [a.name for a in outcome.accepted] == [
        "f0.txt", "f1.txt", "f2.txt", "f3.txt", "f4.txt",
    ]
    assert any("5" in notice and "/reset" in notice
               for notice in outcome.notices)
    # Los descartados no llegan al almacén.
    assert set(await store.list(KEY)) == {"f0.txt", "f1.txt", "f2.txt", "f3.txt", "f4.txt"}


async def test_exactamente_cinco_archivos_no_genera_aviso_de_exceso():
    service, _ = make_service()
    items = [(incoming(f"f{i}.txt"), b"contenido") for i in range(5)]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert len(outcome.accepted) == 5
    assert all("siguiente interacción" not in notice for notice in outcome.notices)


async def test_orden_de_llegada_conserva_los_primeros():
    service, _ = make_service()
    items = [(incoming(f"orden-{i}.txt"), b"x") for i in range(6)]

    outcome = await service.intake(KEY, items, request_text=None)

    assert [a.name for a in outcome.accepted] == [
        f"orden-{i}.txt" for i in range(5)
    ]


# --- Tarea 3.2: peso máximo y fallo de descarga ----------------------------

async def test_archivo_sobre_20mb_se_rechaza_con_aviso():
    service, store = make_service()
    grande = incoming("video.mp4", size=20 * 1024 * 1024 + 1)
    items = [(grande, b"x"), (incoming("ok.txt"), b"contenido")]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert "video.mp4" in outcome.rejected
    assert any("20 MB" in notice for notice in outcome.notices)
    assert await store.list(KEY) == ("ok.txt",)


async def test_archivo_de_exactamente_20mb_se_acepta():
    service, store = make_service()
    items = [(incoming("grande.txt", size=20 * 1024 * 1024), b"x")]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert outcome.rejected == ()
    assert await store.list(KEY) == ("grande.txt",)


async def test_fallo_de_descarga_es_aislado_y_avisa():
    service, store = make_service()
    items = [
        (incoming("roto.txt"), None),
        (incoming("ok.txt"), b"contenido"),
    ]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert "roto.txt" in outcome.rejected
    assert any("No pude descargar" in notice for notice in outcome.notices)
    # El resto sigue procesándose.
    assert [a.name for a in outcome.accepted] == ["ok.txt"]
    assert await store.list(KEY) == ("ok.txt",)


async def test_bytes_reales_sobre_20mb_se_rechazan_sin_tamano_declarado():
    """Req 1.3: el límite aplica también si Telegram no declara `file_size`."""
    service, store = make_service()
    items = [(incoming("sin-tamano.bin", size=0), b"x" * (20 * 1024 * 1024 + 1))]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert "sin-tamano.bin" in outcome.rejected
    assert await store.list(KEY) == ()


# --- Tarea 3.3: clasificación y no procesables ------------------------------

def make_service_with_extractor(
    binary_ok: set[str] | None = None,
) -> tuple[AttachmentService, FakeAttachmentStore]:
    return make_service(binary_ok=binary_ok)


async def test_binario_soportado_por_el_proveedor_se_envia_en_binario():
    service, store = make_service_with_extractor(binary_ok={"foto.jpg"})
    items = [(incoming("foto.jpg"), b"\xff\xd8")]

    outcome = await service.intake(KEY, items, request_text="describe")

    assert outcome.accepted[0].mode is AttachmentMode.BINARY
    assert await store.list(KEY) == ("foto.jpg",)


async def test_foto_sin_mime_se_detecta_por_firma_y_se_guarda_como_imagen():
    store = FakeAttachmentStore()
    service = AttachmentService(
        store=store,
        binary_supported=lambda name, mime: mime.startswith("image/"),
        clock=FakeClock(),
        extract_text=None,
    )
    photo = IncomingAttachment(
        file_id="telegram-photo", file_name=None, mime_type=None,
        file_size=4, kind="photo",
    )

    outcome = await service.intake(KEY, [(photo, b"\xff\xd8\xff\xdb")], request_text="describe")

    assert outcome.action is NextAction.PROCESS
    assert outcome.accepted == (
        StoredAttachment(
            name="archivo-telegram-photo.jpg",
            mime_type="image/jpeg",
            mode=AttachmentMode.BINARY,
            size=4,
        ),
    )


@pytest.mark.parametrize(
    ("image_bytes", "expected_mime", "expected_extension"),
    (
        (b"\x89PNG\r\n\x1a\n\x00", "image/png", ".png"),
        (b"GIF87a", "image/gif", ".gif"),
        (b"RIFF\x00\x00\x00\x00WEBP", "image/webp", ".webp"),
    ),
)
async def test_foto_sin_mime_detecta_formato_por_firma(
    image_bytes: bytes, expected_mime: str, expected_extension: str,
):
    service, _ = make_service(binary_ok={f"archivo-photo{expected_extension}"})
    photo = IncomingAttachment(
        file_id="photo", file_name=None, mime_type=None,
        file_size=len(image_bytes), kind="photo",
    )

    outcome = await service.intake(KEY, [(photo, image_bytes)], request_text="describe")

    assert outcome.accepted[0].mime_type == expected_mime
    assert outcome.accepted[0].name == f"archivo-photo{expected_extension}"


async def test_foto_con_firma_desconocida_no_se_interpreta_como_texto():
    service, store = make_service_with_extractor()
    photo = IncomingAttachment(
        file_id="unknown-photo", file_name=None, mime_type=None,
        file_size=7, kind="photo",
    )

    outcome = await service.intake(KEY, [(photo, b"not-img")], request_text="describe")

    assert outcome.accepted == ()
    assert outcome.rejected == ("archivo-unknown-photo",)
    assert await store.list(KEY) == ()


async def test_audio_adjunto_se_acepta_para_transcripcion_aunque_ai_no_acepte_binarios():
    service, store = make_service()
    audio = IncomingAttachment(
        file_id="audio-1", file_name="nota.mp3", mime_type="audio/mpeg",
        file_size=8, kind="audio",
    )

    outcome = await service.intake(KEY, [(audio, b"ID3audio")], request_text="transcribe")

    assert outcome.action is NextAction.PROCESS
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].mode is AttachmentMode.BINARY
    assert outcome.accepted[0].mime_type == "audio/mpeg"
    assert outcome.accepted[0].name.startswith("audio-adjunto-")
    assert "audio-1" not in outcome.accepted[0].name
    assert await store.load(KEY, outcome.accepted[0].name) == b"ID3audio"


async def test_document_mp3_con_mime_generico_se_clasifica_por_firma():
    service, store = make_service()
    audio = IncomingAttachment(
        file_id="telegram-file-id", file_name=None,
        mime_type="application/octet-stream", file_size=8, kind="document",
    )

    outcome = await service.intake(KEY, [(audio, b"ID3audio")], request_text="resume lo hablado")

    assert outcome.action is NextAction.PROCESS
    assert len(outcome.accepted) == 1
    stored = outcome.accepted[0]
    assert stored.mime_type == "audio/mpeg"
    assert stored.name.endswith(".mp3")
    assert "telegram-file-id" not in stored.name
    assert await store.load(KEY, stored.name) == b"ID3audio"


async def test_nombre_mp3_sin_contenido_audio_no_se_clasifica_como_audio():
    service, store = make_service()
    document = IncomingAttachment(
        file_id="telegram-file-id", file_name="engaño.mp3",
        mime_type="application/octet-stream", file_size=8, kind="document",
    )

    outcome = await service.intake(KEY, [(document, b"not-audio")], request_text="transcribe")

    assert outcome.accepted == ()
    assert outcome.rejected == ("engaño.mp3",)
    assert await store.list(KEY) == ()


async def test_nota_de_voz_no_usa_el_flujo_de_adjunto_pendiente_ni_expone_file_id():
    service, store = make_service()
    voice = IncomingAttachment(
        file_id="opaque-voice-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )

    outcome = await service.intake(KEY, [(voice, b"OggSdata")], request_text=None)

    assert outcome.accepted == ()
    assert outcome.action is NextAction.ASK_INTENT
    assert "opaque-voice-id" not in " ".join(outcome.notices)
    assert await store.list(KEY) == ()


async def test_audio_sin_caption_se_guarda_pendiente_y_pide_instrucciones():
    service, store = make_service()
    audio = IncomingAttachment(
        file_id="audio-1", file_name=None, mime_type=None,
        file_size=8, kind="audio",
    )

    outcome = await service.intake(KEY, [(audio, b"OggSdata")], request_text=None)

    assert outcome.action is NextAction.ASK_INSTRUCTIONS
    assert outcome.has_valid_attachments is True
    assert len(outcome.accepted) == 1
    assert outcome.accepted[0].name.startswith("audio-adjunto-")
    assert "audio-1" not in outcome.accepted[0].name
    assert await store.load(KEY, outcome.accepted[0].name) == b"OggSdata"
    payloads = await service.payloads(KEY)
    assert len(payloads) == 1
    assert payloads[0].mode is AttachmentMode.BINARY
    assert payloads[0].mime_type == "audio/ogg"
    assert payloads[0].data == b"OggSdata"


async def test_nombres_de_audios_pendientes_no_colisionan():
    service, store = make_service()
    first = IncomingAttachment(
        file_id="audio-1", file_name=None, mime_type="audio/mpeg",
        file_size=4, kind="audio",
    )
    second = IncomingAttachment(
        file_id="audio-2", file_name=None, mime_type="audio/mpeg",
        file_size=4, kind="audio",
    )

    outcome = await service.intake(
        KEY, [(first, b"ID3a"), (second, b"ID3b")], request_text=None,
    )

    names = tuple(item.name for item in outcome.accepted)
    assert outcome.action is NextAction.ASK_INSTRUCTIONS
    assert names == ("audio-adjunto-1.mp3", "audio-adjunto-2.mp3")
    assert await store.load(KEY, names[0]) == b"ID3a"
    assert await store.load(KEY, names[1]) == b"ID3b"


async def test_audio_que_excede_limite_se_rechaza_sin_file_id():
    service, store = make_service()
    audio = IncomingAttachment(
        file_id="opaque-audio-id", file_name=None, mime_type="audio/ogg",
        file_size=MAX_FILE_BYTES + 1, kind="audio",
    )

    outcome = await service.intake(KEY, [(audio, None)], request_text="transcribe")

    assert outcome.accepted == ()
    assert outcome.rejected == ("audio adjunto",)
    assert any("20 MB" in notice for notice in outcome.notices)
    assert "opaque-audio-id" not in " ".join(outcome.notices)
    assert await store.list(KEY) == ()


async def test_adjuntos_de_miembros_distintos_del_grupo_quedan_aislados():
    service, store = make_service_with_extractor()
    ana = AttachmentKey(kind=ChatKind.GROUP, chat_id=-100, user_id=7)
    luis = AttachmentKey(kind=ChatKind.GROUP, chat_id=-100, user_id=8)

    await service.intake(ana, [(incoming("nota.txt"), b"Ana")], request_text=None)
    await service.intake(luis, [(incoming("nota.txt"), b"Luis")], request_text=None)

    ana_payloads = await service.payloads(ana)
    luis_payloads = await service.payloads(luis)

    assert ana_payloads[0].text == "Ana"
    assert luis_payloads[0].text == "Luis"
    assert ana.value != luis.value
    assert await store.list(ana) == ("nota.txt",)
    assert await store.list(luis) == ("nota.txt",)


async def test_texto_y_pdf_se_extraen_como_texto():
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    from io import BytesIO

    buffer = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.write(buffer)

    service, store = make_service_with_extractor()
    items = [
        (incoming("nota.txt"), b"hola"),
        (incoming("doc.pdf"), buffer.getvalue()),
    ]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert [a.mode for a in outcome.accepted] == [AttachmentMode.TEXT, AttachmentMode.TEXT]


async def test_tipo_no_procesable_se_elimina_avisa_y_pregunta_intencion():
    service, store = make_service_with_extractor(binary_ok={"foto.jpg"})
    items = [
        (incoming("foto.jpg"), b"\xff\xd8"),
        (incoming("datos.zip"), b"PK"),
    ]

    outcome = await service.intake(KEY, items, request_text="resume")

    assert "datos.zip" in outcome.rejected
    assert any("datos.zip" in notice for notice in outcome.notices)
    assert outcome.has_valid_attachments
    assert outcome.action is NextAction.ASK_INTENT
    # El no procesable no queda en el almacén.
    assert await store.list(KEY) == ("foto.jpg",)


# --- Tarea 3.4: intención pendiente ----------------------------------------

async def test_confirmar_intencion_procede_con_los_validos():
    service, store = make_service_with_extractor(binary_ok={"foto.jpg"})
    items = [
        (incoming("foto.jpg"), b"\xff\xd8"),
        (incoming("datos.zip"), b"PK"),
    ]
    outcome = await service.intake(KEY, items, request_text="resume las fotos")
    assert outcome.action is NextAction.ASK_INTENT

    decision = await service.resolve_intent(KEY, "procede")

    assert decision.action is NextAction.PROCESS
    assert decision.request == "resume las fotos"


async def test_negacion_mantiene_la_espera():
    service, _ = make_service_with_extractor(binary_ok={"foto.jpg"})
    await service.intake(
        KEY,
        [(incoming("foto.jpg"), b"\xff\xd8"), (incoming("datos.zip"), b"PK")],
        request_text=None,
    )

    decision = await service.resolve_intent(KEY, "no")

    assert decision.action is NextAction.ASK_INTENT


async def test_adjuntar_mas_archivos_acumula_en_la_sesion():
    service, store = make_service_with_extractor(binary_ok={"foto.jpg"})
    await service.intake(KEY, [(incoming("foto.jpg"), b"\xff\xd8")], request_text=None)

    decision = await service.resolve_intent(KEY, "procede")
    assert decision.action is NextAction.PROCESS

    # Siguiente interacción: se acumulan con los anteriores.
    outcome = await service.intake(KEY, [(incoming("foto.jpg"), b"\xff\xd8")],
                                   request_text="ahora esta")
    assert outcome.action is NextAction.PROCESS
    assert set(await store.list(KEY)) == {"foto.jpg"}


async def test_la_acumulacion_de_adjuntos_no_supera_el_maximo_por_solicitud():
    service, store = make_service()

    await service.intake(
        KEY,
        [(incoming(f"primero-{index}.txt"), b"x") for index in range(4)],
        request_text=None,
    )
    outcome = await service.intake(
        KEY,
        [(incoming("quinto.txt"), b"x"), (incoming("sexto.txt"), b"x")],
        request_text=None,
    )

    assert len(await service.payloads(KEY)) == 5
    assert await store.list(KEY) == (
        "primero-0.txt", "primero-1.txt", "primero-2.txt", "primero-3.txt",
        "quinto.txt",
    )
    assert any("/reset" in notice for notice in outcome.notices)


async def test_sin_texto_se_pide_instruccion():
    service, _ = make_service_with_extractor()

    outcome = await service.intake(KEY, [(incoming("nota.txt"), b"hola")],
                                   request_text=None)

    assert outcome.action is NextAction.ASK_INSTRUCTIONS
    assert service.is_awaiting(KEY)

    decision = await service.resolve_intent(KEY, "resume el texto")
    assert decision.action is NextAction.PROCESS
    assert decision.request == "resume el texto"


async def test_tras_confirmar_el_estado_queda_listo_para_payloads():
    from oraculo.domain.attachments import AttachmentPayload

    service, _ = make_service_with_extractor(binary_ok={"foto.jpg"})
    await service.intake(KEY, [
        (incoming("foto.jpg"), b"\xff\xd8"),
        (incoming("datos.zip"), b"PK"),
    ], request_text="describe")

    await service.resolve_intent(KEY, "procede")
    payloads = await service.payloads(KEY)

    assert [p.name for p in payloads] == ["foto.jpg"]
    assert all(isinstance(p, AttachmentPayload) for p in payloads)


# --- Tarea 3.5: limpieza en reset, caducidad y arranque --------------------

async def test_reset_elimina_archivos_y_estado_pendiente():
    service, store = make_service_with_extractor(binary_ok={"foto.jpg"})
    await service.intake(KEY, [(incoming("datos.zip"), b"PK")], request_text="resume")

    await service.reset(KEY)

    assert await store.list(KEY) == ()
    # Sin estado pendiente: una nueva interacción no hereda la intención.
    decision = await service.resolve_intent(KEY, "procede")
    assert decision.action is NextAction.ASK_INSTRUCTIONS


async def test_startup_vacia_todo_el_almacen():
    service, store = make_service_with_extractor()
    await service.intake(KEY, [(incoming("nota.txt"), b"hola")], request_text="resume")

    await service.startup()

    assert store.cleared_all == 1
    assert store.files == {}


async def test_caducidad_por_inactividad_elimina_adjuntos_de_la_sesion():
    from oraculo.application.attachments import is_stale

    clock = FakeClock()
    from oraculo.infrastructure.text_extractor import TextExtractor

    store = FakeAttachmentStore()
    service = AttachmentService(
        store=store,
        binary_supported=lambda name, mime: False,
        clock=clock,
        extract_text=TextExtractor(max_chars=1000).extract,
    )
    started = clock()
    await service.intake(KEY, [(incoming("nota.txt"), b"hola")], request_text="resume")

    assert not is_stale(started, clock(), minutes=240)
    clock.advance_minutes(241)
    assert is_stale(started, clock(), minutes=240)


async def test_expire_if_stale_borra_adjuntos_por_inactividad():
    """Req 3.3: la caducidad por inactividad elimina los adjuntos."""
    from oraculo.infrastructure.text_extractor import TextExtractor

    clock = FakeClock()
    store = FakeAttachmentStore()
    service = AttachmentService(
        store=store,
        binary_supported=lambda name, mime: False,
        clock=clock,
        extract_text=TextExtractor(max_chars=1000).extract,
        idle_minutes=240,
    )
    await service.intake(KEY, [(incoming("nota.txt"), b"hola")], request_text="resume")

    assert await service.expire_if_stale(KEY) is False
    assert store.files

    clock.advance_minutes(241)
    assert await service.expire_if_stale(KEY) is True
    assert store.files == {}
    assert service.is_awaiting(KEY) is False


async def test_expire_if_stale_no_borra_si_hay_actividad():
    """Cada uso de los adjuntos renueva la actividad de la sesión."""
    from oraculo.infrastructure.text_extractor import TextExtractor

    clock = FakeClock()
    store = FakeAttachmentStore()
    service = AttachmentService(
        store=store,
        binary_supported=lambda name, mime: False,
        clock=clock,
        extract_text=TextExtractor(max_chars=1000).extract,
        idle_minutes=240,
    )
    await service.intake(KEY, [(incoming("nota.txt"), b"hola")], request_text="resume")
    clock.advance_minutes(200)
    await service.payloads(KEY)
    clock.advance_minutes(200)

    assert await service.expire_if_stale(KEY) is False
    assert store.files


# --- Tarea 4.2: acceso a adjuntos (Req 4.1, 4.2) ---------------------------

def test_privado_no_autorizado_no_admite_adjuntos():
    from oraculo.application.attachments import attachments_allowed
    from oraculo.domain.models import ChatKind, IncomingMessage

    message = IncomingMessage(
        chat_id=1, user_id=99, chat_type=ChatKind.PRIVATE, text=None,
        mentions_bot=False, replies_to_bot=False, quoted_text=None,
    )

    assert not attachments_allowed(message, allowed_users=frozenset({7}), allowed_chats=frozenset())


def test_privado_autorizado_admite_adjuntos():
    from oraculo.application.attachments import attachments_allowed
    from oraculo.domain.models import ChatKind, IncomingMessage

    message = IncomingMessage(
        chat_id=7, user_id=7, chat_type=ChatKind.PRIVATE, text=None,
        mentions_bot=False, replies_to_bot=False, quoted_text=None,
    )

    assert attachments_allowed(message, allowed_users=frozenset({7}), allowed_chats=frozenset())


def test_grupo_sin_mencion_ni_respuesta_se_ignora():
    from oraculo.application.attachments import attachments_allowed
    from oraculo.domain.models import ChatKind, IncomingMessage

    message = IncomingMessage(
        chat_id=1, user_id=99, chat_type=ChatKind.GROUP, text=None,
        mentions_bot=False, replies_to_bot=False, quoted_text=None,
    )

    assert not attachments_allowed(message, allowed_users=frozenset(), allowed_chats=frozenset({1}))


def test_grupo_autorizado_con_mencion_admite_adjuntos():
    from oraculo.application.attachments import attachments_allowed
    from oraculo.domain.models import ChatKind, IncomingMessage

    message = IncomingMessage(
        chat_id=1, user_id=99, chat_type=ChatKind.GROUP, text=None,
        mentions_bot=True, replies_to_bot=False, quoted_text=None,
    )

    assert attachments_allowed(message, allowed_users=frozenset(), allowed_chats=frozenset({1}))


def test_grupo_no_autorizado_se_ignora_incluso_con_mencion():
    from oraculo.application.attachments import attachments_allowed
    from oraculo.domain.models import ChatKind, IncomingMessage

    message = IncomingMessage(
        chat_id=2, user_id=99, chat_type=ChatKind.GROUP, text=None,
        mentions_bot=True, replies_to_bot=False, quoted_text=None,
    )

    assert not attachments_allowed(message, allowed_users=frozenset(), allowed_chats=frozenset({1}))


# --- Tarea 6.2: avisos al usuario y ausencia de secretos (RNF-1) -----------

async def test_los_avisos_de_limites_son_comprensibles():
    service, _ = make_service_with_extractor()
    items = [
        (incoming("enorme.mp4", size=21 * 1024 * 1024), b"x"),
        *[(incoming(f"f{i}.txt"), b"x") for i in range(6)],
    ]

    outcome = await service.intake(KEY, items, request_text="resume")

    notices = "\n\n".join(outcome.notices)
    assert "/reset" in notices
    assert "20 MB" in notices
    # Ningún aviso expone rutas internas ni tokens.
    assert "temp/" not in notices
    assert "sk-" not in notices


def test_errores_de_adjunto_no_contienen_secretos():
    from oraculo.domain.errors import AttachmentError

    for reason in ("too_large", "download_failed", "unsupported", "not_found"):
        error = AttachmentError(reason)
        text = str(error)
        assert "sk-" not in text
        assert "token" not in text.lower()
        assert "Bearer" not in text


async def test_el_log_de_descarga_fallida_no_expone_el_token(caplog):
    import logging

    from telegram.error import TelegramError
    from oraculo.telegram.attachments import download_attachment

    class ExplodingBot:
        def get_file(self, file_id):
            raise TelegramError("fallo con Bearer sk-SECRETO")

    attachment = incoming("a.txt")
    with caplog.at_level(logging.WARNING):
        assert await download_attachment(ExplodingBot(), attachment) is None

    assert "sk-SECRETO" not in caplog.text


# --- Tarea 6.3 [opcional]: telemetría de eventos de adjuntos ---------------

async def test_intake_registra_eventos_en_el_log(caplog):
    import logging

    service, _ = make_service_with_extractor(binary_ok={"foto.jpg"})
    items = [
        (incoming("foto.jpg"), b"\xff\xd8"),
        (incoming("datos.zip"), b"PK"),
    ]

    with caplog.at_level(logging.INFO, logger="oraculo.application.attachments"):
        await service.intake(KEY, items, request_text="resume")

    assert any("adjunto" in record.message.lower() for record in caplog.records)
    # Sin datos del usuario en el log.
    assert all("PK" not in record.message for record in caplog.records)
