"""Tests de integración de adjuntos en `ConversationService` (Wave 4, tarea 4.3)."""

from __future__ import annotations

from datetime import datetime, timezone

from fakes import (
    FakeAIClient,
    FakeAccessPolicy,
    FakeClock,
    FakePhraseProvider,
    FakeSessionRepository,
    group_message,
    private_message,
)

from oraculo.application.attachments import AttachmentService
from oraculo.application.conversation import ConversationService
from oraculo.application.sessions import SessionManager
from oraculo.application.skills import SkillService
from oraculo.domain.attachments import AttachmentMode, IncomingAttachment
from oraculo.domain.errors import AIError, AudioTranscriptionError
from oraculo.domain.models import AttachmentKey, ChatKind, SkillCatalogSnapshot
from oraculo.domain.ports import AudioTranscriber
from oraculo.infrastructure.text_extractor import TextExtractor
from test_attachment_service import FakeAttachmentStore

NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def incoming(name: str, size: int = 10) -> IncomingAttachment:
    return IncomingAttachment(
        file_id=f"file-{name}", file_name=name,
        mime_type="application/octet-stream", file_size=size, kind="document",
    )


def build(
    allowed: bool = True,
    budget: int | None = None,
    attachment_clock=None,
    allowed_chats: frozenset[int] = frozenset(),
    audio_transcriber: AudioTranscriber | None = None,
):
    ai = FakeAIClient(reply="respuesta con archivos")
    store = FakeAttachmentStore()
    attachment_service = AttachmentService(
        store=store,
        binary_supported=lambda name, mime: name.endswith(".jpg"),
        clock=attachment_clock or (lambda: NOW),
        extract_text=TextExtractor(max_chars=1000).extract,
    )
    sessions = SessionManager(
        repository=FakeSessionRepository(), max_turns=15, idle_minutes=240,
        clock=FakeClock(NOW),
    )
    skill_service = _skills()
    extra = {"attachment_budget_bytes": budget} if budget is not None else {}
    service_args = dict(
        ai_client=ai,
        access_policy=FakeAccessPolicy(
            frozenset({1}) if allowed else frozenset(), allowed_chats
        ),
        phrase_provider=FakePhraseProvider(),
        sessions=sessions,
        skill_service=skill_service,
        attachment_service=attachment_service,
    )
    if audio_transcriber is None:
        service = ConversationService(**service_args, **extra)
    else:
        service = ConversationService(
            **service_args, audio_transcriber=audio_transcriber, **extra,
        )
    return service, ai, store, attachment_service


class FakeAudioTranscriber:
    def __init__(
        self,
        transcript: str = "Mañana salimos a las ocho.",
        error: AudioTranscriptionError | None = None,
    ) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    async def transcribe(self, data: bytes, *, mime_type: str) -> str:
        self.calls.append((data, mime_type))
        if self.error is not None:
            raise self.error
        return self.transcript


def _skills() -> SkillService:
    return SkillService(
        SkillCatalogSnapshot(),
        autoload_enabled=False,
        catalog_max_chars=100,
        max_context_chars=100,
        max_per_turn=1,
        max_rounds=1,
    )


async def test_interaccion_con_adjuntos_y_texto_procesa_con_el_llm():
    service, ai, store, _ = build()
    items = [(incoming("nota.txt"), b"hola")]

    reply = await service.handle_attachments(
        private_message(text="resume"), items,
    )

    assert reply is not None
    assert ai.calls  # llegó al LLM
    assert ai.attachments_calls[-1] and ai.attachments_calls[-1][0].name == "nota.txt"
    assert await store.list(service._attachment_key(private_message(text="resume"))) == ()


async def test_foto_telegram_sin_mime_llega_al_llm_con_mime_resuelto():
    service, ai, _, _ = build()
    photo = IncomingAttachment(
        file_id="telegram-photo", file_name=None, mime_type=None,
        file_size=4, kind="photo",
    )

    reply = await service.handle_attachments(
        private_message(text="describe la imagen"), [(photo, b"\xff\xd8\xff\xdb")],
    )

    assert reply is not None
    assert len(ai.calls) == 1
    payload = ai.attachments_calls[0][0]
    assert payload.mime_type == "image/jpeg"
    assert payload.name == "archivo-telegram-photo.jpg"
    assert ai.calls[0][-1].content == "describe la imagen"


async def test_adjunto_incompatible_se_informa_sin_llamar_al_llm():
    service, ai, store, _ = build()

    reply = await service.handle_attachments(
        private_message(text="analiza este archivo"),
        [(incoming("datos.zip"), b"PK")],
    )

    assert reply is not None
    assert "No puedo procesar" in reply
    assert "Adjunta archivos compatibles" in reply
    assert not ai.calls
    assert store.files == {}


async def test_voice_se_transcribe_como_turno_y_no_envia_binario_al_llm():
    transcriber = FakeAudioTranscriber()
    service, ai, store, _ = build(audio_transcriber=transcriber)
    voice = IncomingAttachment(
        file_id="voice-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )

    reply = await service.handle_attachments(
        private_message(text=None), [(voice, b"OggSdata")],
    )

    assert reply == "respuesta con archivos"
    assert transcriber.calls == [(b"OggSdata", "audio/ogg")]
    assert len(ai.calls) == 1
    assert "Transcripción de la nota de voz" in ai.calls[0][-1].content
    assert "Mañana salimos a las ocho." in ai.calls[0][-1].content
    assert ai.attachments_calls[-1] == ()
    assert store.files == {}


async def test_caption_de_voice_se_conserva_junto_a_la_transcripcion():
    transcriber = FakeAudioTranscriber()
    service, ai, _, _ = build(audio_transcriber=transcriber)
    voice = IncomingAttachment(
        file_id="voice-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )

    await service.handle_attachments(
        private_message(text="Resume lo que digo"), [(voice, b"OggSdata")],
    )

    prompt = ai.calls[0][-1].content
    assert "Resume lo que digo" in prompt
    assert "Mañana salimos a las ocho." in prompt


async def test_voice_no_autorizada_no_invoca_transcriptor_ni_llm():
    transcriber = FakeAudioTranscriber()
    service, ai, _, _ = build(allowed=False, audio_transcriber=transcriber)
    voice = IncomingAttachment(
        file_id="voice-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )

    reply = await service.handle_attachments(
        private_message(text=None), [(voice, b"OggSdata")],
    )

    assert reply == "No hablo con extraños"
    assert transcriber.calls == []
    assert ai.calls == []


async def test_voice_en_grupo_solo_se_transcribe_tras_mencion_o_respuesta():
    transcriber = FakeAudioTranscriber()
    service, ai, _, _ = build(
        allowed_chats=frozenset({-100}), audio_transcriber=transcriber,
    )
    voice = IncomingAttachment(
        file_id="voice-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )
    incoming_group = group_message(
        text=None, chat_id=-100, mentions_bot=False, replies_to_bot=True,
    )

    reply = await service.handle_attachments(incoming_group, [(voice, b"OggSdata")])

    assert reply == "respuesta con archivos"
    assert len(transcriber.calls) == 1
    assert len(ai.calls) == 1


async def test_audio_en_grupo_sin_activacion_no_se_transcribe():
    transcriber = FakeAudioTranscriber()
    service, ai, _, _ = build(
        allowed_chats=frozenset({-100}), audio_transcriber=transcriber,
    )
    audio = IncomingAttachment(
        file_id="audio-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="audio",
    )

    reply = await service.handle_attachments(
        group_message(
            text="resume", chat_id=-100, mentions_bot=False,
            replies_to_bot=False,
        ),
        [(audio, b"OggSdata")],
    )

    assert reply is None
    assert transcriber.calls == []
    assert ai.calls == []


async def test_voice_sin_transcriptor_configurado_no_llama_al_llm():
    service, ai, _, _ = build()
    voice = IncomingAttachment(
        file_id="voice-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )

    reply = await service.handle_attachments(
        private_message(text=None), [(voice, b"OggSdata")],
    )

    assert reply == "El procesamiento de audio no está configurado en este bot."
    assert ai.calls == []


async def test_voice_sin_transcripcion_utilizable_no_llama_al_llm():
    transcriber = FakeAudioTranscriber(transcript="  ")
    service, ai, _, _ = build(audio_transcriber=transcriber)
    voice = IncomingAttachment(
        file_id="voice-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="voice",
    )

    reply = await service.handle_attachments(
        private_message(text=None), [(voice, b"OggSdata")],
    )

    assert reply == "No detecté voz hablada en la nota de voz."
    assert len(transcriber.calls) == 1
    assert ai.calls == []


async def test_audio_adjunto_con_caption_se_transcribe_y_envia_solo_texto_al_llm():
    transcriber = FakeAudioTranscriber()
    service, ai, store, _ = build(audio_transcriber=transcriber)
    audio = IncomingAttachment(
        file_id="audio-file-id", file_name=None, mime_type="audio/mpeg",
        file_size=8, kind="audio",
    )

    reply = await service.handle_attachments(
        private_message(text="Resume lo que se dice"), [(audio, b"ID3audio")],
    )

    assert reply == "respuesta con archivos"
    assert transcriber.calls == [(b"ID3audio", "audio/mpeg")]
    assert ai.calls[0][-1].content == "Resume lo que se dice"
    payload = ai.attachments_calls[0][0]
    assert payload.mode is AttachmentMode.TEXT
    assert payload.data is None
    assert "Mañana salimos a las ocho." in payload.text
    assert "no se proporcionó el audio original" in payload.text.lower()
    assert "sonidos no verbales" in payload.text
    assert store.files == {}


async def test_audio_sin_caption_espera_instruccion_antes_de_asr_y_llm():
    transcriber = FakeAudioTranscriber()
    service, ai, store, _ = build(audio_transcriber=transcriber)
    audio = IncomingAttachment(
        file_id="audio-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="audio",
    )
    key = service._attachment_key(private_message(text=None))

    reply = await service.handle_attachments(
        private_message(text=None), [(audio, b"OggSdata")],
    )

    assert reply is not None and "¿Qué quieres hacer" in reply
    assert transcriber.calls == []
    assert ai.calls == []
    assert await store.list(key) != ()

    reply = await service.handle(private_message(text="Resume lo que se dice"))

    assert reply == "respuesta con archivos"
    assert transcriber.calls == [(b"OggSdata", "audio/ogg")]
    assert ai.calls[0][-1].content == "Resume lo que se dice"
    assert await store.list(key) == ()
    await service.handle(private_message(text="¿Qué más puedes decirme?"))
    assert len(transcriber.calls) == 1
    assert ai.attachments_calls[-1] == ()


async def test_audio_pendiente_de_un_miembro_no_se_reutiliza_por_otro():
    transcriber = FakeAudioTranscriber()
    service, ai, _, _ = build(
        allowed_chats=frozenset({-100}), audio_transcriber=transcriber,
    )
    audio = IncomingAttachment(
        file_id="audio-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="audio",
    )

    await service.handle_attachments(
        group_message(
            text=None, chat_id=-100, user_id=1, mentions_bot=False,
            replies_to_bot=True,
        ),
        [(audio, b"OggSdata")],
    )
    await service.handle(group_message(
        text="transcribe", chat_id=-100, user_id=2, mentions_bot=False,
        replies_to_bot=True,
    ))

    assert transcriber.calls == []
    assert ai.calls
    assert ai.attachments_calls[-1] == ()


async def test_fallo_al_transcribir_audio_pendiente_limpia_binario_y_no_llama_al_llm():
    transcriber = FakeAudioTranscriber(error=AudioTranscriptionError("provider_rejected"))
    service, ai, store, _ = build(audio_transcriber=transcriber)
    audio = IncomingAttachment(
        file_id="audio-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="audio",
    )
    key = service._attachment_key(private_message(text=None))

    await service.handle_attachments(private_message(text=None), [(audio, b"OggSdata")])
    reply = await service.handle(private_message(text="transcribe"))

    assert reply is not None and "No pude transcribir" in reply
    assert "provider_rejected" not in reply
    assert ai.calls == []
    assert await store.list(key) == ()


async def test_fallo_del_llm_despues_de_transcribir_limpia_audio_temporal():
    transcriber = FakeAudioTranscriber()
    service, ai, store, _ = build(audio_transcriber=transcriber)
    ai.error = AIError("provider timeout")
    audio = IncomingAttachment(
        file_id="audio-file-id", file_name=None, mime_type="audio/ogg",
        file_size=8, kind="audio",
    )
    key = service._attachment_key(private_message(text="resume lo dicho"))

    reply = await service.handle_attachments(
        private_message(text="resume lo dicho"), [(audio, b"OggSdata")],
    )

    assert reply is not None and "provider timeout" in reply
    assert len(transcriber.calls) == 1
    assert ai.calls
    assert await store.list(key) == ()


async def test_rechazos_preguntan_intencion_sin_llamar_al_llm():
    service, ai, store, _ = build()
    items = [
        (incoming("foto.jpg"), b"\xff\xd8"),
        (incoming("datos.zip"), b"PK"),
    ]

    reply = await service.handle_attachments(private_message(text="resume"), items)

    assert reply is not None
    assert "datos.zip" in reply
    assert not ai.calls  # nada se procesa hasta conocer la intención


async def test_confirmar_intencion_procesa_con_los_validos():
    service, ai, store, _ = build()
    items = [
        (incoming("foto.jpg"), b"\xff\xd8"),
        (incoming("datos.zip"), b"PK"),
    ]
    await service.handle_attachments(private_message(text="resume las fotos"), items)

    reply = await service.handle(private_message(text="procede"))

    assert reply is not None
    assert ai.calls
    assert [p.name for p in ai.attachments_calls[-1]] == ["foto.jpg"]


async def test_reset_borra_los_adjuntos_de_la_sesion():
    service, ai, store, attachment_service = build()
    await service.handle_attachments(
        private_message(text="resume"), [(incoming("nota.txt"), b"hola")],
    )

    reply = await service.handle_reset(private_message(text="/new"))

    assert reply is not None
    assert store.files == {}


async def test_sin_acceso_no_se_almacena_nada():
    service, ai, store, _ = build(allowed=False)
    items = [(incoming("nota.txt"), b"hola")]

    await service.handle_attachments(private_message(text="resume"), items)

    assert not store.files


# --- Tarea 6.1: punta a punta con reutilización posterior ------------------

async def test_los_adjuntos_se_consumen_tras_procesarse_correctamente():
    """Req R4: adjuntos procesados con éxito no se reutilizan en otro turno."""
    service, ai, store, _ = build()

    await service.handle_attachments(
        private_message(text="resume el documento"),
        [(incoming("informe.txt"), b"datos del informe")],
    )
    assert ai.calls
    assert store.files == {}

    # Turno posterior de solo texto: no hereda el adjunto consumido.
    ai.calls.clear()
    ai.attachments_calls.clear()
    reply = await service.handle(private_message(text="¿qué decía el informe?"))

    assert reply is not None
    assert ai.calls
    assert ai.attachments_calls[-1] == ()
    assert await store.list(_key()) == ()


async def test_fallo_del_llm_conserva_adjuntos_para_reintento_y_exito_los_consume():
    service, ai, store, _ = build()
    ai.error = AIError("provider_error", user_message="fallo temporal")

    failed = await service.handle_attachments(
        private_message(text="resume el archivo"),
        [(incoming("nota.txt"), b"contenido")],
    )

    assert failed is not None and "fallo temporal" in failed
    assert await store.list(_key()) == ("nota.txt",)
    ai.error = None
    ai.attachments_calls.clear()

    retried = await service.handle(private_message(text="intenta de nuevo"))

    assert retried is not None
    assert [payload.name for payload in ai.attachments_calls[-1]] == ["nota.txt"]
    assert await store.list(_key()) == ()


async def test_adjunto_pendiente_de_un_miembro_no_se_expone_a_otro_en_grupo():
    service, ai, store, _ = build(allowed_chats=frozenset({-100}))
    ana = group_message(text=None, user_id=1, replies_to_bot=True)

    prompt = await service.handle_attachments(
        ana, [(incoming("nota.txt"), b"solo Ana")],
    )

    assert prompt is not None and "Qué quieres hacer" in prompt
    assert not ai.calls
    await service.handle(group_message(text="resume", user_id=2, replies_to_bot=True))
    assert ai.attachments_calls[-1] == ()
    assert await store.list(_attachment_key(user_id=1)) == ("nota.txt",)

    await service.handle(group_message(text="resume", user_id=1, replies_to_bot=True))

    assert ai.attachments_calls[-1][0].text == "solo Ana"
    assert await store.list(_attachment_key(user_id=1)) == ()


async def test_texto_sin_adjuntos_no_crea_estado_temporal_vacio():
    service, _, _, attachment_service = build()

    await service.handle(private_message(text="solo texto"))

    assert attachment_service._states == {}


async def test_reset_de_otro_miembro_no_borra_adjuntos_pendientes():
    service, _, store, _ = build(allowed_chats=frozenset({-100}))
    await service.handle_attachments(
        group_message(text=None, user_id=1, replies_to_bot=True),
        [(incoming("nota.txt"), b"solo Ana")],
    )

    await service.handle_reset(group_message(text="/reset", user_id=2))

    assert await store.list(_attachment_key(user_id=1)) == ("nota.txt",)


async def test_flujo_completo_rechazo_confirmacion_y_procesado():
    """Punta a punta: rechazo → intención → confirmación → respuesta."""
    service, ai, store, _ = build()

    reply1 = await service.handle_attachments(
        private_message(text="analiza los archivos"),
        [
            (incoming("foto.jpg"), b"\xff\xd8"),
            (incoming("video.avi"), b"RIFF"),
        ],
    )
    assert reply1 is not None
    assert "video.avi" in reply1
    assert not ai.calls

    reply2 = await service.handle(private_message(text="procede"))
    assert reply2 is not None
    assert ai.calls
    assert [p.name for p in ai.attachments_calls[-1]] == ["foto.jpg"]


async def test_adjuntos_que_exceden_presupuesto_no_se_envian_parcialmente():
    """Req R1/R5: no dividir los archivos en llamadas que pierdan contexto."""
    service, ai, store, _ = build(budget=1)
    items = [
        (incoming("a.txt"), b"aaaa"),
        (incoming("b.txt"), b"bbbb"),
    ]

    reply = await service.handle_attachments(private_message(text="resume"), items)

    assert reply is not None
    assert not ai.calls
    assert "Usa /reset" in reply
    assert set(await store.list(_key())) == {"a.txt", "b.txt"}


async def test_texto_y_todos_los_adjuntos_compatibles_viajan_en_una_llamada():
    service, ai, _, _ = build()

    reply = await service.handle_attachments(
        private_message(text="compara los documentos"),
        [
            (incoming("a.txt"), b"primero"),
            (incoming("b.txt"), b"segundo"),
        ],
    )

    assert reply is not None
    assert len(ai.calls) == 1
    assert ai.calls[0][-1].content == "compara los documentos"
    assert [payload.name for payload in ai.attachments_calls[0]] == ["a.txt", "b.txt"]


async def test_adjuntos_caducan_con_la_inactividad_de_la_sesion():
    """Req 3.3: sin actividad, los adjuntos se eliminan antes del turno."""

    class MovingClock:
        def __init__(self) -> None:
            self.now = NOW

        def __call__(self):
            return self.now

        def advance_hours(self, hours: int) -> None:
            from datetime import timedelta

            self.now += timedelta(hours=hours)

    clock = MovingClock()
    service, ai, store, _ = build(attachment_clock=clock)
    await service.handle_attachments(
        private_message(text="resume"), [(incoming("nota.txt"), b"hola")],
    )
    assert ai.calls

    clock.advance_hours(5)
    ai.calls.clear()
    ai.attachments_calls.clear()
    reply = await service.handle(private_message(text="¿qué decía?"))

    assert reply is not None
    assert ai.calls
    assert ai.attachments_calls[-1] == ()
    assert store.files == {}


def _key():
    return AttachmentKey(kind=ChatKind.PRIVATE, chat_id=100, user_id=1)


def _attachment_key(*, user_id: int) -> AttachmentKey:
    return AttachmentKey(kind=ChatKind.GROUP, chat_id=-100, user_id=user_id)
