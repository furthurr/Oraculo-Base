"""Pruebas del adaptador local de transcripción con el modelo simulado."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
import threading
import time
from typing import BinaryIO

import pytest

from oraculo.domain.errors import AudioTranscriptionError
from oraculo.infrastructure.audio_transcription import FasterWhisperTranscriber


@dataclass(frozen=True)
class FakeSegment:
    text: str


class FakeWhisperModel:
    def __init__(
        self,
        model_size: str,
        *,
        device: str,
        compute_type: str,
        segment_texts: tuple[str, ...] = (" Hola", " desde la nota. "),
        failure: Exception | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self.options = (model_size, device, compute_type)
        self.segment_texts = segment_texts
        self.failure = failure
        self.delay_s = delay_s
        self.inputs: list[BinaryIO] = []
        self.languages: list[str | None] = []
        self.thread_ids: list[int] = []
        self.calls = 0
        self.active_calls = 0
        self.max_active_calls = 0
        self._counter_lock = threading.Lock()

    def transcribe(
        self,
        audio: BinaryIO,
        *,
        beam_size: int = 5,
        language: str | None = None,
    ) -> tuple[Iterable[FakeSegment], object]:
        del beam_size
        if self.failure is not None:
            raise self.failure
        self.inputs.append(audio)
        self.languages.append(language)
        self.calls += 1
        with self._counter_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.thread_ids.append(threading.get_ident())

        def segments() -> Iterable[FakeSegment]:
            try:
                if self.delay_s:
                    time.sleep(self.delay_s)
                yield from (FakeSegment(text) for text in self.segment_texts)
            finally:
                with self._counter_lock:
                    self.active_calls -= 1

        return segments(), object()


def install_fake_model(
    monkeypatch: pytest.MonkeyPatch,
    *,
    segment_texts: tuple[str, ...] = (" Hola", " desde la nota. "),
    failure: Exception | None = None,
    delay_s: float = 0.0,
) -> list[FakeWhisperModel]:
    models: list[FakeWhisperModel] = []

    def create_model(
        model_size: str,
        *,
        device: str,
        compute_type: str,
    ) -> FakeWhisperModel:
        model = FakeWhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            segment_texts=segment_texts,
            failure=failure,
            delay_s=delay_s,
        )
        models.append(model)
        return model

    monkeypatch.setattr("faster_whisper.WhisperModel", create_model)
    return models


async def test_transcribe_usa_model_base_local_y_concatena_segmentos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = install_fake_model(monkeypatch)
    transcriber = FasterWhisperTranscriber()

    transcript = await transcriber.transcribe(b"voice bytes", mime_type="audio/ogg")

    assert transcript == "Hola desde la nota."
    assert models[0].options == ("base", "cpu", "int8")
    assert isinstance(models[0].inputs[0], BytesIO)
    assert models[0].inputs[0].getvalue() == b"voice bytes"
    assert models[0].languages == ["es"]


async def test_modelo_se_carga_perezosamente_una_vez_y_se_reutiliza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = install_fake_model(monkeypatch)
    transcriber = FasterWhisperTranscriber()

    assert models == []
    await transcriber.transcribe(b"first", mime_type="audio/ogg")
    await transcriber.transcribe(b"second", mime_type="audio/ogg")

    assert len(models) == 1
    assert models[0].calls == 2


async def test_modelo_y_segmentos_se_procesan_fuera_del_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = install_fake_model(monkeypatch)
    event_loop_thread = threading.get_ident()

    await FasterWhisperTranscriber().transcribe(b"audio", mime_type="audio/ogg")

    assert models[0].thread_ids
    assert models[0].thread_ids[0] != event_loop_thread


async def test_transcript_vacio_falla_de_forma_tipada(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_model(monkeypatch, segment_texts=("  ", ""))

    with pytest.raises(AudioTranscriptionError) as excinfo:
        await FasterWhisperTranscriber().transcribe(b"audio", mime_type="audio/ogg")

    assert excinfo.value.reason == "empty_transcript"


async def test_entrada_vacia_o_mime_no_admitido_no_carga_modelo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = install_fake_model(monkeypatch)
    transcriber = FasterWhisperTranscriber()

    with pytest.raises(AudioTranscriptionError) as empty:
        await transcriber.transcribe(b"", mime_type="audio/ogg")
    with pytest.raises(AudioTranscriptionError) as unsupported:
        await transcriber.transcribe(b"audio", mime_type="application/octet-stream")
    with pytest.raises(AudioTranscriptionError) as unknown_audio:
        await transcriber.transcribe(b"audio", mime_type="audio/x-unknown")

    assert empty.value.reason == "empty_audio"
    assert unsupported.value.reason == "unsupported_format"
    assert unknown_audio.value.reason == "unsupported_format"
    assert models == []


async def test_fallo_del_modelo_se_redacta_en_error_y_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret_detail = "private-model-path"
    install_fake_model(monkeypatch, failure=RuntimeError(secret_detail))

    with pytest.raises(AudioTranscriptionError) as excinfo:
        await FasterWhisperTranscriber().transcribe(b"audio", mime_type="audio/ogg")

    assert excinfo.value.reason == "transcription_failed"
    assert secret_detail not in str(excinfo.value)
    assert secret_detail not in caplog.text


async def test_solicitudes_concurrentes_reutilizan_modelo_y_se_serializan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = install_fake_model(monkeypatch, delay_s=0.02)
    transcriber = FasterWhisperTranscriber()

    transcripts = await asyncio.gather(*(
        transcriber.transcribe(f"audio-{index}".encode(), mime_type="audio/ogg")
        for index in range(3)
    ))

    assert transcripts == ["Hola desde la nota."] * 3
    assert len(models) == 1
    assert models[0].calls == 3
    assert models[0].max_active_calls == 1
